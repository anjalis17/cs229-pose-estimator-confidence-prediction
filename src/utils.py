# src/utils.py
# Contains utility functions to load SPEED+ ground truth labels and camera intrinsics 
# from JSON files

import json
import numpy as np
from pathlib import Path
from src.errors import quaternion_to_rotation_matrix


def load_labels(json_path: str) -> list[dict]:
    """
    Load SPEED+ JSON label file (ground truth poses for each image)
    Returns list of dicts with keys:
        filename, R_true (3x3), t_true (3,)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels = []
    for entry in data:
        q = np.array(entry['q_vbs2tango_true'])
        t = np.array(entry['r_Vo2To_vbs_true'])
        labels.append({
            'filename': entry['filename'],
            'R_true': quaternion_to_rotation_matrix(q),
            't_true': t
        })
    return labels


def load_camera(camera_json_path: str) -> dict:
    """
    Load camera intrinsics from camera.json.
    """
    with open(camera_json_path, 'r') as f:
        return json.load(f)