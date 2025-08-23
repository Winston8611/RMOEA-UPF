# RMOEA_UPF_Project/selection_history.py

import numpy as np
from .utils import cosine_distance, non_dominated_sort
from .upf import compute_usps_for_solution

def final_solution_selection(archive_infos, k, problem):
    """
    Implements the final solution selection based purely on the historical
    evaluation data contained within the archive. 
    Args:
        archive_infos (list): The final archive from the RMOEA-UPF run.
        k (int): The desired number of final solutions.
        problem (object): The problem instance, used for objective space dimensionality.

    Returns:
        list: A list of k selected solutions.
    """
    if not archive_infos:
        print("Archive is empty. Cannot perform final selection.")
        return []
        
    # Filter out any solutions that might not have a valid 'obj' due to early termination.
    valid_archive_infos = [sol for sol in archive_infos if sol.get('obj') is not None]
    if len(valid_archive_infos) != len(archive_infos):
        print(f"Warning: Filtered out {len(archive_infos) - len(valid_archive_infos)} solutions with invalid objective values.")
    
    if not valid_archive_infos:
        print("No valid solutions left after filtering. Cannot perform final selection.")
        return []

    # If the archive is already small enough, return all valid solutions.
    if len(valid_archive_infos) <= k:
        print("Archive size is smaller than or equal to k, returning all valid solutions.")
        return valid_archive_infos

    print("Starting final solution selection (History-Based Mode)...")

    # --- Step 1: Calculate the true USP for each solution from its history ---
    print("  - Calculating USPs from solution histories...")
    all_final_usps_flat = []
    usp_to_sol_map = []
    for i, sol in enumerate(valid_archive_infos):
        # Use historical data to calculate a reliable USP set.
        if 'history_objs' in sol and len(sol['history_objs']) > 1:
            history_array = np.array(sol['history_objs'])
            usps = compute_usps_for_solution(history_array, alpha=0.9)
            sol['final_usps'] = usps
        else:
            sol['final_usps'] = np.array([sol['obj']])
        
        # Collect all USPs for a global non-dominated sort.
        if sol['final_usps'].size > 0:
            for usp in sol['final_usps']:
                all_final_usps_flat.append(usp)
                usp_to_sol_map.append(i)

    if not all_final_usps_flat:
        print("Warning: No USPs generated from historical data. Returning a random subset.")
        selected_indices = np.random.choice(len(valid_archive_infos), k, replace=False)
        return [valid_archive_infos[i] for i in selected_indices]

    # --- Step 2: Determine the hierarchical rank for each solution ---
    print("  - Determining hierarchical ranks for all solutions...")
    # Perform a global sort on all USPs to establish fronts.
    usp_fronts = non_dominated_sort(np.array(all_final_usps_flat))
    usp_rank_map = {}
    for rank_idx, front in enumerate(usp_fronts):
        for usp_idx_in_flat_array in front:
            usp_tuple = tuple(all_final_usps_flat[usp_idx_in_flat_array])
            usp_rank_map[usp_tuple] = rank_idx

    # Assign a rank to each solution based on its USPs.
    for sol in valid_archive_infos:
        if 'final_usps' in sol and sol['final_usps'].size > 0:
            ranks_of_my_usps = [usp_rank_map.get(tuple(usp), float('inf')) for usp in sol['final_usps']]
            # Primary criterion: the best (lowest) rank achieved by any of its USPs.
            best_rank = min(ranks_of_my_usps)
            # Secondary criterion (tie-breaker): the number of USPs on that best front.
            count_in_best_rank = sum(1 for rank in ranks_of_my_usps if rank == best_rank)
            sol['rank'] = (best_rank, -count_in_best_rank) 
        else:
            sol['rank'] = (float('inf'), 0)

    # --- Step 3: Use reference vectors for diversity-based selection ---
    print("  - Performing reference vector-based selection...")
    if problem.n_obj == 2:
        angles = np.linspace(0, np.pi / 2, k)
        reference_vectors = np.array([np.cos(angles), np.sin(angles)]).T
    else: 
        rand_vecs = np.random.rand(k, problem.n_obj)
        reference_vectors = rand_vecs / np.linalg.norm(rand_vecs, axis=1)[:, np.newaxis]

    # Associate each solution with the closest reference vector.
    ref_vec_candidates = {i: [] for i in range(k)}
    all_objs = np.array([sol['obj'] for sol in valid_archive_infos])
    obj_ideal = np.min(all_objs, axis=0)
    
    for sol in valid_archive_infos:
        normalized_obj = sol['obj'] - obj_ideal
        distances = [cosine_distance(normalized_obj, ref_vec) for ref_vec in reference_vectors]
        closest_ref_idx = np.argmin(distances)
        ref_vec_candidates[closest_ref_idx].append(sol)

    # For each reference vector, select the best-ranked candidate.
    final_solutions_infos = []
    selected_solutions_set = set()

    for ref_idx in range(k):
        candidates = ref_vec_candidates[ref_idx]
        if candidates:
            best_candidate = min(candidates, key=lambda x: x.get('rank', (float('inf'), 0)))
            solution_id = tuple(best_candidate['dec'])
            if solution_id not in selected_solutions_set:
                final_solutions_infos.append(best_candidate)
                selected_solutions_set.add(solution_id)

    # --- Step 4: Fill any remaining slots if the selection is smaller than k ---
    if len(final_solutions_infos) < k:
        print(f"  - Only {len(final_solutions_infos)} solutions selected. Filling remaining slots...")
        # Create a pool of solutions that were not selected in the first pass.
        all_solutions_ids = {tuple(sol['dec']) for sol in valid_archive_infos}
        loser_solutions_ids = all_solutions_ids - selected_solutions_set
        loser_solutions_pool = [sol for sol in valid_archive_infos if tuple(sol['dec']) in loser_solutions_ids]
        
        # Sort the remaining solutions by their rank and add the best ones.
        loser_solutions_pool.sort(key=lambda x: x.get('rank', (float('inf'), 0)))
        
        remaining_slots = k - len(final_solutions_infos)
        final_solutions_infos.extend(loser_solutions_pool[:remaining_slots])

    print(f"Final selection chose {len(final_solutions_infos)} solutions.")
    return final_solutions_infos
