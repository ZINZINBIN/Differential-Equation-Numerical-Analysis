import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from typing import List, Union, Literal, Callable

# random nodes
def generate_random_nodes(n_nodes : int, lower : float, upper : float):
    result = np.random.rand(n_nodes)
    result = lower + (upper - lower) * result
    return result

# uniform nodes
def generate_uniform_nodes(n_nodes : int, lower : float, upper : float):
    result = np.array([i / n_nodes for i in range(n_nodes)])
    result = lower + (upper - lower) * result
    return result

# chebyshev nodes
def generate_chebyshev_nodes(n_nodes : int, lower : float, upper : float, mode_type : Literal['type_1', 'type_2'] = 'type_1'):
    if mode_type == 'type_1':
        indices = [math.pi * idx / n_nodes for idx in range(0,n_nodes)]
    else:
        indices = [math.pi * (2 * idx + 1) / (2 * n_nodes + 1) for idx in range(0,n_nodes)]
    result = (lower + upper)/2  + (upper - lower)/2 * np.cos(indices)
    return sorted(result)

def func(x : float)->float:
    return 1.0 / (1.0 + 16 * x ** 2)

# compute polynomial 
# p_k(x) = a_0 + a_1 * (x-x_0) + a_2 * (x-x_0) * (x-x_1) + ....
def compute_polynomial(x : float, coeffs : List, x_data : List):
    n = len(coeffs)
    result = coeffs[0]
    dx = 1

    for i in range(1,n):
        dx *= (x - x_data[i-1])
        result += coeffs[i] * dx
        
    return result

# compute polynomial with multiple data
# x : [x_0, x_1, ...]
# output : [p_k(x_0), p_k(x_1), ...]
def compute_polynomials(xs : Union[List, np.array], coeffs : List, x_data : List):
    y = [] 
    for x in xs:
        y.append(compute_polynomial(x, coeffs, x_data))
    return np.array(y)

# compute Newton interpolation coefficient
def compute_coeffs(func : Callable[[float], float], x_data : Union[List[float], np.array]):
    y_data = [func(x) for x in x_data]
    y_data = np.array(y_data)
        
    mat = np.zeros((len(x_data), len(x_data)))
    mat[:,0] = y_data

    coeffs = []   

    for idx_i in range(1, len(x_data)):
        dys = mat[:,idx_i-1]
        for idx_j in range(idx_i, len(x_data)):
            dx = x_data[idx_j] - x_data[idx_j-idx_i]
            dy = dys[idx_j] - dys[idx_j-1]
            mat[idx_j, idx_i] = dy / dx
    
    coeffs = np.diag(mat)
    return coeffs

if __name__ == "__main__":

    n_nodes = 8
    lower = -1.0
    upper = 1.0

    nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper, 'type_2')
    nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

    coeffs_cheb = compute_coeffs(func, nodes_cheb)
    coeffs_uniform = compute_coeffs(func, nodes_uniform)

    xs = generate_uniform_nodes(32, lower, upper)
 
    y_cheb = compute_polynomials(xs, coeffs_cheb, nodes_cheb)
    y_uniform = compute_polynomials(xs, coeffs_uniform, nodes_uniform)
    y_real = np.array([func(x) for x in xs])

    plt.figure(figsize = (8,6))
    plt.plot(xs, y_real, 'ro-',label = '1 /(1 + x^2)')
    plt.plot(xs, y_cheb, 'bo-',label = 'interpolate with chebshev')
    plt.plot(xs, y_uniform, 'ko-',label = 'interpolate with uniform')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.savefig("./interpolation.png")