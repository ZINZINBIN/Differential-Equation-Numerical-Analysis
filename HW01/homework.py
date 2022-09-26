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
from typing import List, Callable, Union

# nodes generator : random / uniform / chebshev
def generate_random_nodes(n_nodes : int, lower : float, upper : float):
    result = np.random.rand(n_nodes)
    result = lower + (upper - lower) * result
    return result

def generate_uniform_nodes(n_nodes : int, lower : float, upper : float):
    result = np.array([i / n_nodes for i in range(n_nodes)])
    result = lower + (upper - lower) * result
    return result

def generate_chebyshev_nodes(n_nodes : int, lower : float, upper : float):
    indices = [math.pi * idx / n_nodes for idx in range(0,n_nodes)]
    result = (lower + upper)/2  + (upper - lower)/2 * np.cos(indices)
    return sorted(result)

# function : y = sin(pi * x)
def func(x : float, lower : float = 0, upper : float = 2.0)->float:
    if x >= lower and x <=upper:
        return np.sin(math.pi * x)
    else:
        ValueError("x should be in range of [{},{}]".format(lower, upper))

def compute_polynomial(x : float, coeffs : List, x_data : List):
    n = len(coeffs)
    result = coeffs[0]
    dx = 1

    for i in range(1,n):
        dx *= (x - x_data[i-1])
        result += coeffs[i] * dx
        
    return result

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
    lower = 0
    upper = 2

    nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper)
    nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

    coeffs_cheb = compute_coeffs(func, nodes_cheb)
    coeffs_uniform = compute_coeffs(func, nodes_uniform)

    print("coffs_cheb : ", coeffs_cheb)
    print("coffs_uniform : ", coeffs_uniform)

    xs = generate_uniform_nodes(16, 0, 2)
    y_cheb = []
    y_uniform = []
    for x in xs:
        y_cheb.append(compute_polynomial(x, coeffs_cheb, coeffs_cheb))
        y_uniform.append(compute_polynomial(x, coeffs_uniform, coeffs_uniform))
        
    y_cheb = np.array(y_cheb)
    y_uniform = np.array(y_uniform)
    y_real = np.array([func(x) for x in xs])

    plt.figure(figsize = (8,6))
    plt.plot(xs, y_real, 'ro-',label = 'sin(pi * x)')
    plt.plot(xs, y_cheb, 'bo-',label = 'interpolate with chebshev')
    plt.plot(xs, y_uniform, 'ko-',label = 'interpolate with uniform')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.savefig("./interpolation.png")
