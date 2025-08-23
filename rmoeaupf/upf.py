# RMOEA_UPF_Project/upf.py
import numpy as np
from .utils import non_dominated_sort

def compute_usps_for_solution(noisy_objs, alpha):
    """
    Calculates all Uncertain a-Support Points (USPs) for a single solution
    based on its performance under noise. This implementation is vectorized for efficiency.
    
    Args:
        noisy_objs (np.ndarray): An array of shape (n_noise_samples, n_obj) containing
                                 the objective values of a solution under multiple noise perturbations.
        alpha (float): The confidence level.
        
    Returns:
        np.ndarray: An array containing one or more USP vectors.
    """
    # Ensure input is always a 2D array to handle single evaluation history.
    if noisy_objs.ndim == 1:
        noisy_objs = noisy_objs.reshape(1, -1)

    n_noise_samples = noisy_objs.shape[0]
    if n_noise_samples == 0:
        return np.array([])

    # Use unique observed objective vectors as candidate points to reduce computation.
    candidate_points = np.unique(noisy_objs, axis=0)

    # --- Vectorized calculation of the domination matrix ---
    # This is a highly efficient way to check dominance for all pairs of
    # candidate points against all noisy objective points.
    # all_le[i, j] is True if candidate_points[i] <= noisy_objs[j] in all objectives.
    all_le = np.all(candidate_points[:, np.newaxis, :] <= noisy_objs[np.newaxis, :, :], axis=2)
    # any_lt[i, j] is True if candidate_points[i] < noisy_objs[j] in at least one objective.
    any_lt = np.any(candidate_points[:, np.newaxis, :] < noisy_objs[np.newaxis, :, :], axis=2)
    domination_matrix = all_le & any_lt

    # Calculate the probability that each candidate point dominates a random noisy observation.
    domination_counts = np.sum(domination_matrix, axis=1)
    probs = domination_counts / n_noise_samples

    # --- Select USPs based on the confidence level alpha ---
    # A point is a potential USP if its domination probability is less than or equal to (1 - alpha).
    target_prob = 1 - alpha
    eligible_mask = probs <= target_prob
    
    # Fallback strategy: if no points meet the criteria, return the worst-performing point.
    # This ensures the algorithm always has a point to consider.
    if not np.any(eligible_mask):
        worst_point_idx = np.argmax(np.sum(noisy_objs, axis=1))
        return noisy_objs[worst_point_idx].reshape(1, -1)
    
    # From the eligible candidates, select the one(s) whose probability is closest to the target.
    eligible_candidates = candidate_points[eligible_mask]
    eligible_probs = probs[eligible_mask]
    
    dists = np.abs(eligible_probs - target_prob)
    min_dist = np.min(dists)
    
    best_usps_mask = dists == min_dist
    final_usps = eligible_candidates[best_usps_mask]

    return final_usps


def compute_upf(all_usps_list):
    """
    Computes the Uncertainty-related Pareto Front (UPF) from a list of USP sets
    belonging to different solutions.
    """
    if not all_usps_list:
        return [], np.array([])
        
    valid_usps_list = [usps for usps in all_usps_list if usps.size > 0]
    if not valid_usps_list:
        return [], np.array([])
        
    # Flatten the list of all USP sets into a single array.
    flat_usps = np.vstack(valid_usps_list)
    
    # The UPF is simply the first non-dominated front of all collected USPs.
    fronts = non_dominated_sort(flat_usps)
    if not fronts:
        return [], np.array([])
        
    upf_indices = fronts[0]
    return upf_indices, flat_usps[upf_indices]
