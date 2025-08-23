# RMOEA_UPF_Project/utils.py

import numpy as np

def non_dominated_sort(objs):
    """
    Performs a non-dominated sort on a set of objective vectors.
    This is a standard procedure in multi-objective optimization.
    
    Args:
        objs (np.ndarray): An array of shape (n_points, n_obj) containing objective vectors.
        
    Returns:
        list: A list of lists, where each inner list contains the indices
              of the solutions belonging to a particular non-dominated front.
    """
    n_points = objs.shape[0]
    fronts = [[]]
    
    # domination_counts[i] stores the number of solutions that dominate solution i.
    domination_counts = np.zeros(n_points)
    # dominated_solutions[i] is a list of solutions dominated by solution i.
    dominated_solutions = [[] for _ in range(n_points)]
    
    # Compare all pairs of solutions to build dominance relationships.
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Check if solution i dominates solution j.
            dominates_j = np.all(objs[i, :] <= objs[j, :]) and np.any(objs[i, :] < objs[j, :])
            # Check if solution j dominates solution i.
            dominates_i = np.all(objs[j, :] <= objs[i, :]) and np.any(objs[j, :] < objs[i, :])
            
            if dominates_j:
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif dominates_i:
                dominated_solutions[j].append(i)
                domination_counts[i] += 1
                
    # The first front consists of all solutions with a domination count of 0.
    for i in range(n_points):
        if domination_counts[i] == 0:
            fronts[0].append(i)
            
    # Iteratively build subsequent fronts.
    front_idx = 0
    while len(fronts[front_idx]) > 0:
        next_front = []
        # For each solution in the current front...
        for i in fronts[front_idx]:
            # ...go through all solutions it dominates.
            for j in dominated_solutions[i]:
                # Decrement their domination count.
                domination_counts[j] -= 1
                # If a solution's domination count becomes 0, it belongs to the next front.
                if domination_counts[j] == 0:
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)
        
    # Remove the last empty front.
    fronts.pop()
    return fronts

def cosine_distance(a, b):
    """
    Calculates the cosine distance between two vectors.
    Used for associating solutions to reference vectors.
    """
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def crowding_distance_assignment(objs):
    """
    Calculates the crowding distance for a set of objective vectors.
    Used as a secondary sorting criterion to promote diversity.
    
    Args:
        objs (np.ndarray): An array of shape (n_points, n_obj) of objective vectors,
                         assumed to be on the same non-dominated front.
                         
    Returns:
        np.ndarray: An array of shape (n_points,) containing the crowding distance score for each point.
    """
    n_points, n_obj = objs.shape
    if n_points <= 2:
        # For fronts with 2 or fewer points, the distance is infinite.
        return np.full(n_points, np.inf)

    # Initialize distances to zero.
    distances = np.zeros(n_points)
    
    # Calculate distance for each objective dimension.
    for m in range(n_obj):
        # Get the indices that would sort the current objective.
        sorted_indices = np.argsort(objs[:, m])
        
        # Get the sorted objective values.
        sorted_objs_m = objs[sorted_indices, m]
        
        # Assign infinite distance to the boundary points.
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Calculate distance for the interior points.
        # The distance is the normalized difference between the neighbors.
        norm = sorted_objs_m[-1] - sorted_objs_m[0]
        if norm < 1e-9: # Avoid division by zero if all values are the same.
            continue
            
        for i in range(1, n_points - 1):
            distances[sorted_indices[i]] += (sorted_objs_m[i+1] - sorted_objs_m[i-1]) / norm
            
    return distances
