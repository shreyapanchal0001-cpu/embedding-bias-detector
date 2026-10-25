"""Bias metrics computed from embeddings."""
import numpy as np

def association_score(target_vec: np.ndarray,
                       group_a_centroid: np.ndarray,
                       group_b_centroid: np.ndarray) -> float:
    """How much closer is the target concept to group A vs group B?"""
    return float(np.dot(target_vec, group_a_centroid) - np.dot(target_vec, group_b_centroid))

def disparity_percentage(association_scores: list[float]) -> float:
    """Convert raw association scores to a 0-100% bias score."""
    scores = np.array(association_scores)
    mean_abs = np.abs(scores).mean()
    return min(100.0, abs(mean_abs) / 0.3 * 100)