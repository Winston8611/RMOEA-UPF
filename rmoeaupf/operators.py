# RMOEA_UPF_Project/operators.py

import numpy as np

def sbx_crossover(p1, p2, xl, xu, eta_c=20):
    """模拟二进制交叉 (SBX)"""
    c1, c2 = p1.copy(), p2.copy()
    
    for i in range(len(p1)):
        if np.random.rand() < 1: # 交叉概率
            mu = np.random.rand()
            if mu <= 0.5:
                beta = (2 * mu) ** (1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - mu))) ** (1 / (eta_c + 1))
            
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    
    c1 = np.clip(c1, xl, xu)
    c2 = np.clip(c2, xl, xu)
    return c1, c2

def polynomial_mutation(p, xl, xu, eta_m=20):
    """多项式变异"""
    mutated_p = p.copy()
    prob_mut = 1.0 / len(p)
    # prob_mut=1.0
    
    for i in range(len(p)):
        if np.random.rand() < prob_mut:
            mu = np.random.rand()
            if mu < 0.5:
                delta = (2 * mu) ** (1 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - mu)) ** (1 / (eta_m + 1))
            
            mutated_p[i] += delta * (xu[i] - xl[i])
            
    mutated_p = np.clip(mutated_p, xl, xu)
    return mutated_p