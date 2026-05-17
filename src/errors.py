# src/errors.py
# Contains functions to compute prediction errors (rotation, translation, and overall SPEED score).

import numpy as np

def rotation_error_deg(R_pred: np.ndarray, R_true: np.ndarray) -> float:
    """
    Geodesic angle between two rotation matrices, in degrees.
    R_pred, R_true: (3, 3) numpy arrays
    """
    trace = np.trace(R_pred.T @ R_true)
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def translation_error_m(t_pred: np.ndarray, t_true: np.ndarray) -> float:
    """
    Euclidean distance between predicted and true translation, in meters.
    t_pred, t_true: (3,) numpy arrays
    """
    return float(np.linalg.norm(t_pred - t_true))


def speed_score(R_pred: np.ndarray, t_pred: np.ndarray,
                R_true: np.ndarray, t_true: np.ndarray) -> float:
    """
    SPEED score: E_R (radians) + E_T / ||t_true||
    Lower is better.
    """
    # Based on SPNv2 paper: https://arxiv.org/pdf/2203.04275
    E_R = np.radians(rotation_error_deg(R_pred, R_true))
    E_T = translation_error_m(t_pred, t_true)
    return float(E_R + E_T / np.linalg.norm(t_true))


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.
    """
    qw, qx, qy, qz = q / np.linalg.norm(q)  # normalize first
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])