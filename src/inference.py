# # src/inference.py
# import sys
# import json
# import csv
# import random
# import numpy as np
# import torch
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm

# # ── SPNv2 path setup ──────────────────────────────────────────────────────────
# SPNV2_ROOT = Path(__file__).parent.parent / 'spnv2'
# sys.path.insert(0, str(SPNV2_ROOT))
# sys.path.insert(0, str(SPNV2_ROOT / 'core'))
# from config import cfg, update_config
# from nets   import build_spnv2

# from src.errors import (
#     rotation_error_deg, translation_error_m,
#     speed_score, quaternion_to_rotation_matrix
# )

# # ── Config ────────────────────────────────────────────────────────────────────
# PROJECT_ROOT = Path(__file__).parent.parent
# CONFIG_FILE  = PROJECT_ROOT / 'spnv2/experiments/offline_train_full_config_phi3_BN.yaml'
# CHECKPOINT   = PROJECT_ROOT / 'checkpoints/spnv2_efficientnetb3_fullconfig_offline.pth.tar'
# DATA_ROOT    = PROJECT_ROOT / 'data/speedplusv2/speedplus'
# RESULTS_DIR  = PROJECT_ROOT / 'results'
# RESULTS_DIR.mkdir(exist_ok=True)

# N_SAMPLES = 1000
# SEED      = 42

# DOMAINS = {
#     'synthetic': ('validation.json', 'images'),
#     'lightbox':  ('test.json',       'images'),
#     'sunlamp':   ('test.json',       'images'),
# }

# PREPROCESS = transforms.Compose([
#     transforms.Resize((512, 768)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # ── Helpers ───────────────────────────────────────────────────────────────────

# def get_device():
#     if torch.backends.mps.is_available():
#         return torch.device('mps')
#     return torch.device('cpu')


# def load_model(device):
#     update_config(cfg, type('Args', (), {
#         'cfg':  str(CONFIG_FILE),
#         'opts': [
#             'CUDA',  'False', 'FP16', 'False',
#             'DIST.MULTIPROCESSING_DISTRIBUTED', 'False',
#             'DATASET.ROOT',      str(DATA_ROOT.parent),
#             'DATASET.KEYPOINTS', str(SPNV2_ROOT / 'core/utils/models/tangoPoints.mat'),
#             'TEST.MODEL_FILE',   str(CHECKPOINT),
#         ]
#     })())
#     model = build_spnv2(cfg)
#     ckpt  = torch.load(CHECKPOINT, map_location='cpu')
#     model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
#     return model.to(device).eval()


# def load_image(path):
#     return PREPROCESS(Image.open(path).convert('RGB')).unsqueeze(0)


# # Function to convert 6D rotation representation (that SPNv2 outputs) to 3x3 rotation matrix, via Gram-Schmidt process
# def rot6d_to_matrix(v):
#     """6D vector → 3x3 rotation matrix via Gram-Schmidt."""
#     r1 = v[:3] / (np.linalg.norm(v[:3]) + 1e-8)
#     r2 = v[3:] - np.dot(r1, v[3:]) * r1
#     r2 = r2    / (np.linalg.norm(r2)  + 1e-8)
#     return np.stack([r1, r2, np.cross(r1, r2)], axis=1)


# def parse_output(outputs):
#     """Extract rotation (6D) and translation from SPNv2 output.
    
#     outputs[0]: heatmaps (1, 11, H, W)
#     outputs[1]: list [cls_scores, boxes, rotations, translations]
#         cls_scores:   (1, 73656) - confidence for each anchor 
#         boxes:        (1, 73656, 4) - anchor box coordinates (not needed for SPEED+)
#         rotations:    (1, 73656, 6) - 6D rotation representation for each anchor
#         translations: (1, 73656, 3) - translation vector for each anchor
#     """
#     cls_scores   = outputs[1][0]   # (1, 73656)
#     rotations    = outputs[1][2]   # (1, 73656, 6)
#     translations = outputs[1][3]   # (1, 73656, 3)

#     # Pick the anchor with highest classification score
#     best_anchor = cls_scores[0].argmax().item()

#     rot6d  = rotations[0, best_anchor].cpu().float().numpy()    # (6,)
#     t_pred = translations[0, best_anchor].cpu().float().numpy() # (3,)

#     return rot6d, t_pred


# def compute_errors(R_pred, t_pred, R_true, t_true):
#     return {
#         'speed_score': round(speed_score(R_pred, t_pred, R_true, t_true), 6),
#         'E_T':         round(translation_error_m(t_pred, t_true), 6),
#         'E_R':         round(rotation_error_deg(R_pred, R_true), 6),
#     }


# def save_csv(results, domain):
#     path = RESULTS_DIR / f'per_image_errors_{domain}.csv'
#     with open(path, 'w', newline='') as f:
#         writer = csv.DictWriter(
#             f, fieldnames=['filename', 'speed_score', 'E_T', 'E_R'])
#         writer.writeheader()
#         writer.writerows(results)
#     print(f"Saved {len(results)} rows → {path}")

# # ── Inference loop ────────────────────────────────────────────────────────────

# def run_domain(model, domain, device):
#     json_file, img_dir = DOMAINS[domain]
#     labels = json.load(open(DATA_ROOT / domain / json_file))

#     random.seed(SEED)
#     sampled = random.sample(labels, min(N_SAMPLES, len(labels)))
#     print(f"\n{domain}: {len(sampled)} images")

#     first, results = True, []

#     for entry in tqdm(sampled, desc=domain):
#         filename = entry['filename']
#         R_true   = quaternion_to_rotation_matrix(
#                        np.array(entry['q_vbs2tango_true']))
#         t_true   = np.array(entry['r_Vo2To_vbs_true'])

#         try:
#             img = load_image(DATA_ROOT / domain / img_dir / filename).to(device)
#         except Exception as e:
#             print(f"Image load failed: {filename}: {e}"); continue

#         with torch.no_grad():
#             outputs = model(img)

#         if first:
#             first = False
#             print(f"Output length: {len(outputs)}")
#             for i, o in enumerate(outputs):
#                 if isinstance(o, (list, tuple)):
#                     print(f"  [{i}] list len={len(o)}")
#                     for j, oo in enumerate(o):
#                         if hasattr(oo, 'shape'):
#                             print(f"    [{i}][{j}] shape={oo.shape}")
#                 elif isinstance(o, dict):
#                     print(f"  [{i}] dict keys={list(o.keys())}")
#                 elif hasattr(o, 'shape'):
#                     print(f"  [{i}] shape={o.shape}")

#         try:
#             rot6d, t_pred = parse_output(outputs)
#         except Exception as e:
#             print(f"Parse failed: {filename}: {e}"); continue

#         results.append({'filename': filename,
#                         **compute_errors(rot6d_to_matrix(rot6d),
#                                          t_pred, R_true, t_true)})
#     return results

# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     device = get_device()
#     print(f"Device: {device}")
#     model = load_model(device)
#     print(f"Model loaded from {CHECKPOINT.name}")

#     for domain in DOMAINS:
#         results = run_domain(model, domain, device)
#         save_csv(results, domain)

# if __name__ == '__main__':
#     main()