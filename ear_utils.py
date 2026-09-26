"""
Eye Aspect Ratio (EAR) utilities.

EAR formula (Soukupova & Cech, 2016):

        ||p2 - p6|| + ||p3 - p5||
EAR = -----------------------------
              2 * ||p1 - p4||

Where p1..p6 are the 6 (x, y) landmark points around one eye,
ordered as dlib's 68-point model gives them.
"""

import numpy as np
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye_points):
    """
    Compute EAR for a single eye.

    Args:
        eye_points: array-like of 6 (x, y) coordinates, in dlib's
                    canonical order for that eye.

    Returns:
        float EAR value. Lower = more closed.
    """
    eye_points = np.asarray(eye_points)

    vertical_1 = dist.euclidean(eye_points[1], eye_points[5])
    vertical_2 = dist.euclidean(eye_points[2], eye_points[4])
    horizontal = dist.euclidean(eye_points[0], eye_points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def shape_to_np(shape, dtype="int"):
    """Convert a dlib `full_object_detection` shape object to a NumPy array of (x, y)."""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def average_ear(landmarks, left_range, right_range):
    """
    Given the full 68-point landmark array and the index ranges for
    each eye, return (avg_ear, left_eye_pts, right_eye_pts).
    """
    left_eye = landmarks[left_range[0]:left_range[1]]
    right_eye = landmarks[right_range[0]:right_range[1]]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    avg = (left_ear + right_ear) / 2.0
    return avg, left_eye, right_eye
