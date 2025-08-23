# RMOEA_UPF_Project/problem.py

import numpy as np

# --- Creating a unified base class ---
class BaseProblem:
    """
    Base class for all test problems, handling common initializations.
    This ensures a consistent interface for the optimization algorithm.
    """
    def __init__(self, n_var):
        self.n_var = n_var  # Number of decision variables
        self.n_obj = 2      # Number of objectives (fixed to 2 for these problems)
        self.xl = None      # Lower bounds of decision variables
        self.xu = None      # Upper bounds of decision variables

    def evaluate(self, x):
        """
        Public method to evaluate solutions. It ensures the input is correctly shaped
        before calling the internal, problem-specific evaluation method.
        """
        # Ensure input x is a 2D array (n_samples, n_var) for vectorized calculations.
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        objs = self._evaluate(x)
        return objs

    def _evaluate(self, x):
        """
        Problem-specific evaluation logic. This method must be implemented by each subclass.
        """
        raise NotImplementedError("Each problem must implement its own _evaluate method.")

# --- TP9 Benchmark Problem ---
class TP9(BaseProblem):
    """
    Implements the TP9 benchmark problem from Deb and Gupta (2014).
    This problem features a local Pareto front, which can be challenging for algorithms.
    """
    def __init__(self, n_var=5):
        super().__init__(n_var)
        # Define the lower and upper bounds for the decision variables.
        self.xl = np.array([0, 0] + [-1] * (self.n_var - 2))
        self.xu = np.ones(self.n_var)
        
    def _evaluate(self, x):
        """
        Calculates the objective values for the TP9 problem.
        """
        f1 = x[:, 0]
        
        h = 2 - x[:, 0] - 0.8 * np.exp(-((x[:, 0] + x[:, 1] - 0.35) / 0.25)**2) - np.exp(-((x[:, 0] + x[:, 1] - 0.85) / 0.03)**2)
        
        g = 50 * np.sum(x[:, 2:]**2, axis=1)
        
        s = 1 - np.sqrt(x[:, 0])
        
        f2 = h * (g + s)
        
        # Return the objectives as a stacked numpy array.
        return np.vstack([f1, f2]).T
