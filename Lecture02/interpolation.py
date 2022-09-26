import numpy as np
import scipy as sp
from typing import List, Union
from collections.abc import Callable

def generate_random_nodes(n_nodes : int, lower : float, upper : float):
    result = np.random.rand(n_nodes)
    result = lower + (upper - lower) * result
    return result

def generate_uniform_nodes(n_nodes : int, lower : float, upper : float):
    result = [i for i in range(n_nodes)]
    result = lower + (upper - lower) * result
    return result

def generate_chebyshev_nodes(n_nodes : int, lower : float, upper : float):
    idx_max = n_nodes // 2
    indices = [2 * idx + 1 / n_nodes if n_nodes % 2 == 0 else 2 * idx / n_nodes for idx in range(idx_max)]
    result = (lower + upper)/2  + (upper - lower)/2 * np.cos(indices)
    return result

def func(x : float)->float:
    return 1.0 / (1.0 + 16 * x ** 2)

def compute_polynomial(x : float, coeffs : List, x_data : List):
    n = len(coeffs)
    result = 0
    dx = 1
    for i in range(0,n):
        dx *= (x - x_data[i])
        result += coeffs[i] * dx
    return result

def compute_coeff(func : Callable[[float], float], x_data : Union[List[float], np.array]):
    y_data = [func(x) for x in x_data]

    if type(x_data) == np.array:
        y_data = np.array(y_data)
        
    coeffs = []   

    for idx_i in range(0, len(x_data)-1):
        idx_j = idx_i + 1
        coeff = y_data[idx_j] - y_data[idx_i]
        coeff /= (x_data[idx_j] - x_data[idx_i])
        coeffs.append(coeff)

    return coeffs

if __name__ == "__main__":

    n_nodes_lst = [4,8,12,16]
    lower = -1.0
    upper = 1.0

    for n_nodes in n_nodes_lst:
        x_uniform = generate_uniform_nodes(n_nodes, lower, upper)
        x_cheb = generate_chebyshev_nodes(n_nodes, lower, upper)

        coeffs_uniform = compute_coeff(func, x_uniform)
        coeffs_cheb = compute_coeff(func, x_cheb)


    