''' 
Homework01
- editor : JinSu KIM (2019-27420)
- list
(1) Take 4 (x,y) points (y = sin(πx), 0<x<2, 
choose x values arbitrarily) and find the polynomial that interpolates these points. 
Add one more point (obeying the above condition) and find the polynomial that interpolates your 5 points. 
(Choose Handwriting or Word-processing - First part of your report

(2) Make your code (any algorithm is okay) to compute integration of y = sin(πx) from 0 to 2 
and integration of your interpolation polynomials. 
Compare your results while increasing the node number for the integration of y = sin(πx).  
(Second part of your report)

- Explanation
(1) Interpolation : Newton Method
(2) nodes : use Chebshev nodes and uniform nodes
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Callable, Union, Literal

# nodes generator : random / uniform / chebshev
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

# function : y = sin(pi * x)
def func(x : float, lower : float = 0, upper : float = 2.0)->float:
    if x >= lower and x <=upper:
        return np.sin(math.pi * x)
    else:
        ValueError("x should be in range of [{},{}]".format(lower, upper))

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

    n_nodes = 4
    lower = 0
    upper = 2

    nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper, 'type_2')
    nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

    coeffs_cheb = compute_coeffs(func, nodes_cheb)
    coeffs_uniform = compute_coeffs(func, nodes_uniform)

    xs = generate_uniform_nodes(32, lower, upper)
 
    y_cheb = compute_polynomials(xs, coeffs_cheb, nodes_cheb)
    y_uniform = compute_polynomials(xs, coeffs_uniform, nodes_uniform)
    y_real = np.array([func(x) for x in xs])

    plt.figure(figsize = (8,6))
    plt.plot(xs, y_real, 'ro-',label = 'sin(pi * x)')
    plt.plot(xs, y_cheb, 'bo-',label = 'interpolate with chebshev')
    plt.plot(xs, y_uniform, 'ko-',label = 'interpolate with uniform')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.savefig("./interpolation.png")

    # use different n_nodes

    n_nodes_list = [4, 8, 16]
    y_cheb_list = []
    y_uniform_list = []

    for n_nodes in n_nodes_list:
        nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper, 'type_2')
        nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

        coeffs_cheb = compute_coeffs(func, nodes_cheb)
        coeffs_uniform = compute_coeffs(func, nodes_uniform)

        y_cheb = compute_polynomials(xs, coeffs_cheb, nodes_cheb)
        y_uniform = compute_polynomials(xs, coeffs_uniform, nodes_uniform)

        y_cheb_list.append(y_cheb)
        y_uniform_list.append(y_uniform)

    plt.figure(figsize = (8,6))
    plt.plot(xs, y_real, 'ro-',label = 'sin(pi * x)')

    for n_nodes, y_cheb, y_uniform in zip(n_nodes_list, y_cheb_list, y_uniform_list):
        plt.plot(xs, y_cheb, label = "n_nodes : {} cheb".format(n_nodes))
        plt.plot(xs, y_uniform, label = "n_nodes : {} uniform".format(n_nodes))

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.savefig("./interpolation_n_nodes.png")