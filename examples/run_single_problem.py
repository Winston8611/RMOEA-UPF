# examples/run_single_problem.py

import numpy as np
import os

# Assuming the project is structured with a 'rmoeaupf' package
from rmoeaupf.algorithm import RMOEA_UPF
from rmoeaupf.problems import TP9

def main():
    """
    A simple example demonstrating how to use the encapsulated RMOEA_UPF algorithm.
    """
    
    # 1. Define the problem to be solved
    problem = TP9(n_var=10)

    # 2. Set the algorithm parameters
    params = {
        "problem": problem,
        "population_size": 100,
        "archive_capacity": 100,
        "elite_offspring_size": 30,
        "max_real_evals": 2000, # A smaller value for a quick test
        "alpha": 0.9
    }

    # 3. Instantiate and run the algorithm via the simple .solve() method
    print(f"Running RMOEA-UPF on {problem.__class__.__name__}...")
    algorithm = RMOEA_UPF(**params)
    result = algorithm.solve(k_final_solutions=100)

    # 4. Access and save the results from the returned dictionary
    if result["final_solutions"].size > 0:
        output_dir = "./results_output"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"final_solutions_{problem.__class__.__name__}.csv")
        np.savetxt(filepath, result["final_solutions"], delimiter=',')
        
        print(f"\nResults saved to {filepath}")
    else:
        print("Optimization finished, but no final solutions were selected.")

if __name__ == "__main__":
    main()
