"""
Pose error metrics: rotation, translation, and the SPEED score.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import numpy as np

def rotation_error_deg(R_pred, R_true):
    trace = np.trace(R_pred.T @ R_true)
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def translation_error_m(t_pred, t_true):
    return float(np.linalg.norm(t_pred - t_true))


def speed_score(R_pred, t_pred, R_true, t_true):
    # E_R [rad] + E_T / ||t_true||, lower is better (SPNv2 paper, arxiv 2203.04275)
    E_R = np.radians(rotation_error_deg(R_pred, R_true))
    E_T = translation_error_m(t_pred, t_true)
    return float(E_R + E_T / np.linalg.norm(t_true))


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q / np.linalg.norm(q)  # expects [qw, qx, qy, qz]
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
