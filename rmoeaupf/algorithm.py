# rmoeaupf/algorithm.py

import numpy as np
import time
# Assuming these modules are in the same package (rmoeaupf/)
from .operators import sbx_crossover, polynomial_mutation
from .upf import compute_usps_for_solution
from .utils import non_dominated_sort, crowding_distance_assignment
from .selection import final_solution_selection

class RMOEA_UPF:
    """
    RMOEA-UPF (Robust Multi-Objective Evolutionary Algorithm based on UPF).

    This class provides a high-level interface to run the RMOEA-UPF algorithm.
    It encapsulates the initialization, evolutionary loop, and final selection process.
    """
    def __init__(self, problem, population_size=100, archive_capacity=100, elite_offspring_size=20, max_real_evals=10000, alpha=0.9):
        """
        Initializes the RMOEA-UPF algorithm.

        Args:
            problem (object): The problem to be solved, conforming to the BaseProblem interface.
            population_size (int): The number of population in each generation.
            archive_capacity (int): The capacity of the elite archive.
            elite_offspring_size (int): The number of elite offspring to enter the candidate pool.
            max_real_evals (int): The maximum number of real function evaluations.
            alpha (float): The confidence level for USP calculation.
        """
        self.problem = problem
        self.population_size = population_size
        self.archive_capacity = archive_capacity
        self.elite_offspring_size = elite_offspring_size
        self.max_real_evals = max_real_evals
        self.alpha = alpha

        self.archive = []
        self.total_real_evals = 0
        self.result = {}

    def _initialize(self):
        """
        (Internal) Initializes the archive with random solutions.
        """
        print("Initializing and performing initial evaluation...")
        initial_solutions = []
        init_decs = np.random.rand(self.archive_capacity, self.problem.n_var) * (self.problem.xu - self.problem.xl) + self.problem.xl
        
        for i in range(self.archive_capacity):
            if self.total_real_evals >= self.max_real_evals: break
            sol = {'dec': init_decs[i], 'obj': None, 'history_objs': []}
            
            eval_obj = self.problem.evaluate(sol['dec'].reshape(1, -1))[0]
            self.total_real_evals += 1
            sol['history_objs'].append(eval_obj)
            sol['obj'] = eval_obj
            initial_solutions.append(sol)
        
        self.archive = initial_solutions
        print(f"Initialization complete. Initial archive size: {len(self.archive)}")

    def _run_loop(self):
        """
        (Internal) Executes the main evolutionary loop.
        """
        gen = 0
        while self.total_real_evals < self.max_real_evals:
            gen += 1
            print(f"\n--- Generation {gen} | Real Evals: {self.total_real_evals}/{self.max_real_evals} ---")

            # Step 1: Parent Selection
            if not self.archive:
                print("Error: Archive is empty, cannot select parents. Halting.")
                break
            parent_indices = np.random.choice(len(self.archive), self.population_size, replace=True)
            mating_pool = [self.archive[i] for i in parent_indices]

            # Step 2: Reproduction
            offspring_decs = []
            for i in range(self.population_size // 2):
                p1, p2 = np.random.choice(mating_pool, 2, replace=False)
                c1, c2 = sbx_crossover(p1['dec'], p2['dec'], self.problem.xl, self.problem.xu)
                offspring_decs.append(polynomial_mutation(c1, self.problem.xl, self.problem.xu))
                offspring_decs.append(polynomial_mutation(c2, self.problem.xl, self.problem.xu))
            offspring_solutions = [{'dec': dec, 'obj': None, 'history_objs': []} for dec in offspring_decs]

            # Step 3: Offspring Evaluation
            for sol in offspring_solutions:
                if self.total_real_evals >= self.max_real_evals: break
                eval_obj = self.problem.evaluate(sol['dec'].reshape(1, -1))[0]
                self.total_real_evals += 1
                sol['history_objs'].append(eval_obj)
                sol['obj'] = eval_obj
            evaluated_offspring = [s for s in offspring_solutions if s.get('obj') is not None]
            if not evaluated_offspring: continue

            # Step 4: Environmental Selection
            # 4.1. Select elite offspring
            fronts = non_dominated_sort(np.array([s['obj'] for s in evaluated_offspring]))
            top_candidates = []
            for front in fronts:
                top_candidates.extend([evaluated_offspring[i] for i in front])
                if len(top_candidates) >= self.elite_offspring_size: break
            new_entrants = top_candidates[:self.elite_offspring_size]
            
            # 4.2. Re-evaluate candidate pool with noise
            pool = self.archive + new_entrants
            unique_pool = list({tuple(s['dec']): s for s in pool}.values())
            
            noise_range = 0.01 * (self.problem.xu - self.problem.xl)
            for sol in unique_pool:
                if self.total_real_evals >= self.max_real_evals: break
                noise = np.random.uniform(-1, 1, self.problem.n_var) * noise_range
                eval_dec_noisy = np.clip(sol['dec'] + noise, self.problem.xl, self.problem.xu)
                eval_obj_noisy = self.problem.evaluate(eval_dec_noisy.reshape(1, -1))[0]
                self.total_real_evals += 1
                sol['history_objs'].append(eval_obj_noisy)
            
            # 4.3. Rank and truncate archive
            all_usps = []
            for sol in unique_pool:
                usps = compute_usps_for_solution(np.array(sol['history_objs']), self.alpha)
                sol['usps'] = usps if usps.size > 0 else np.atleast_2d(sol['obj'])
                all_usps.extend(list(sol['usps']))
            
            if all_usps:
                usp_fronts = non_dominated_sort(np.array(all_usps))
                usp_rank_map = {tuple(usp): r for r, f in enumerate(usp_fronts) for i in f for usp in [np.array(all_usps)[i]]}
                
                front_groups = {}
                for sol in unique_pool:
                    best_rank = min([usp_rank_map.get(tuple(u), float('inf')) for u in sol['usps']])
                    sol['best_rank'] = best_rank
                    if best_rank not in front_groups: front_groups[best_rank] = []
                    front_groups[best_rank].append(sol)

                for rank, group in front_groups.items():
                    if len(group) > 0:
                        distances = crowding_distance_assignment(np.array([s['obj'] for s in group]))
                        for i, sol in enumerate(group): sol['crowding_dist'] = distances[i]
            
            unique_pool.sort(key=lambda x: (x.get('best_rank', float('inf')), -x.get('crowding_dist', 0)))
            self.archive = unique_pool[:self.archive_capacity]

        print(f"\nAlgorithm finished. Reached max real evaluations: {self.total_real_evals}")
        return self.archive

    def solve(self, k_final_solutions=100):
        """
        The main public method to run the optimization and get the results.

        Args:
            k_final_solutions (int): The number of solutions to select for the final set.

        Returns:
            dict: A dictionary containing the final results.
        """
        start_time = time.time()
        
        self._initialize()
        final_archive = self._run_loop()
        
        final_selected_solutions_info = final_solution_selection(
            final_archive, k_final_solutions, self.problem
        )
        
        end_time = time.time()

        if final_selected_solutions_info:
            final_decs = np.array([s['dec'] for s in final_selected_solutions_info])
            final_objs = np.array([s['obj'] for s in final_selected_solutions_info])
        else:
            final_decs, final_objs = np.array([]), np.array([])

        self.result = {
            "final_solutions": final_decs,
            "final_objectives": final_objs,
            "execution_time_minutes": (end_time - start_time) / 60,
            "final_archive": final_archive
        }
        
        print("\n--- Optimization Complete ---")
        print(f"Obtained {len(final_decs)} final solutions.")
        print(f"Total execution time: {self.result['execution_time_minutes']:.2f} minutes.")
        
        return self.result
