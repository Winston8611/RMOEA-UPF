# RMOEA-UPF: Population-Based Search Method Using Uncertainty-related Pareto Front for Robust Multi-objective Optimization
This repository contains the official Python implementation for the paper: "Population-Based Search Method Using Uncertainty-related Pareto Front for Robust Multi-objective Optimization". RMOEA-UPF is a novel optimization algorithm designed to solve robust multi-objective optimization problems where decision variables are subject to noise.
## Core Concept: The Uncertainty-related Pareto Front (UPF)
Conventional robust multi-objective optimization methods often decouple convergence and robustness, typically treating robustness as a secondary concern. This approach is frequently ineffective, leading to suboptimal solutions in noisy, real-world environments.

Our approach introduces the Uncertainty-related Pareto Front (UPF), a novel framework that fundamentally reframes the problem. Instead of balancing two separate goals, the UPF integrates convergence and robustness into a single, unified objective: a non-dominated front of population-level probabilistic performance guarantees. The RMOEA-UPF algorithm is designed to directly and efficiently optimize this UPF.
## Key Features
- **Co-equally Optimization:** Co-equally optimizes for convergence and robustness by evolving the UPF.
- **Adaptive Sampling:** Efficiently uses a limited budget of real function evaluations by focusing robustness checks on the most promising solutions.
- **Archive-centric approach:** the main population is generated directly from the elite members of the archive. The algorithm's progression is then driven by a comprehensive, evaluation-based environmental selection that determines which of these new candidates will persist and update the archive. 
- **General-Purpose:** Designed to be a general framework applicable to a wide range of robust multi-objective problems.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/RMOEA-UPF.git
   cd RMOEA-UPF
   ```
3. (Recommended) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
5. Install the required dependencies:
   ```
   pip install numpy pandas openpyxl
   ```
## Quick Start
The following example demonstrates how to use the RMOEA_UPF algorithm to solve the TP9 benchmark problem.
```
# examples/run_single_problem.py

import numpy as np
import os
from rmoeaupf.algorithm import RMOEA_UPF
from rmoeaupf.problems import TP9

# 1. Define the problem
problem = TP9(n_var=10)

# 2. Set algorithm parameters
params = {
    "problem": problem,
    "population_size": 100,
    "archive_capacity": 100,
    "elite_offspring_size": 30,
    "max_real_evals": 2000, # Use a smaller value for a quick test
    "alpha": 0.9
}

# 3. Instantiate and run the algorithm
algorithm = RMOEA_UPF(**params)
result = algorithm.solve(k_final_solutions=100)

# 4. Access and save the results
if result["final_solutions"].size > 0:
    os.makedirs("./results_output", exist_ok=True)
    filepath = f"./results_output/final_solutions_{problem.__class__.__name__}.csv"
    np.savetxt(filepath, result["final_solutions"], delimiter=',')
    print(f"\nFinal solutions saved to {filepath}")
```
## How to Run
To avoid `ModuleNotFoundError`, run the example script from the **root directory** of the project (the main `RMOEA-UPF` folder) using the `-m` flag (to treat it as a package):
```
python -m examples.run_single_problem
```
## Citation
If you use this code or the UPF framework in your research, please cite our paper:
```
@article{xu2025population,
  title={Population-Based Search Method Using Uncertainty-related Pareto Front for Robust Multi-objective Optimization},
  author={Xu, Lihong and Jiang, Wenxiang},
  journal={arXiv preprint arXiv:2510.16386},
  year={2025}
}
```




