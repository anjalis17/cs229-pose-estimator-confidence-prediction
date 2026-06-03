"""
Load SPEED+ ground truth labels and camera intrinsics from JSON.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import json
import numpy as np
from pathlib import Path
from src.utils.errors import quaternion_to_rotation_matrix


def load_labels(json_path):
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


def load_camera(camera_json_path):
    with open(camera_json_path, 'r') as f:
        return json.load(f)
