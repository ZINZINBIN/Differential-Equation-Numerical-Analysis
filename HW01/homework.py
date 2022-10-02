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
(1) Interpolation : Newton method (polynomial interpolation)
(2) Nodes : use Chebshev nodes and uniform nodes
(3) Integration : use Simpson / Romberg Algorithm
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Callable, Union, Literal, Optional

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
def func(x : float)->float:
    return np.sin(math.pi * x)

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

# Numerical Integral with Trapezoidal method
def compute_trapezoidal(func : Callable[[float], float], lower : float, upper : float, n_nodes : int):
    
    h = (upper - lower) / n_nodes
    xs = generate_uniform_nodes(n_nodes, lower, upper)
    xs[n_nodes] = upper
    ys = np.array([func(x) for x in xs])

    result = np.sum(ys) + np.sum(ys[1:-1])

    return result * h / 2

# Numerical Integral with Simpson method
def compute_simpson(func : Callable[[float], float], lower : float, upper : float, n_nodes : int):

    assert n_nodes %2 ==0, "n_nodes should be even integer"
    
    h = (upper - lower) / n_nodes
    xs = generate_uniform_nodes(n_nodes, lower, upper)
    np.append(xs, upper)
    ys = np.array([func(x) for x in xs])

    result = ys[0] + 4 * np.sum(ys[1:-1:2]) + 2 * np.sum(ys[0:-1:2]) + ys[-1]

    return result * h / 3

# Trapsezoidal method for recursive rule : compute integral with h = (b-a) / 2 **m
def compute_recursive_trapsezoidal(func : Callable[[float], float], lower : float, upper : float, m : int):
    h = upper - lower
    n_nodes = 1
    T = 0.5 * h  * (func(lower) + func(upper))

    for _ in range(0,m):

        h /= 2
        x_nodes = np.array([lower + h * (2 * idx + 1) for idx in range(0, n_nodes)])
        y_nodes = np.array([func(x) for x in x_nodes])
        n_nodes *= 2

        T = 0.5 * T + h * np.sum(y_nodes)
    
    return T

# compute Romberg triangle R(i,k)
def romberg_algorithm(func : Callable[[float], float], lower : float, upper : float, n : int):

    R = np.zeros((n,n))
    
    for k in range(0,n):
        for i in range(k,n):
            if k == 0:
                # initalize
                R[i,k] = compute_recursive_trapsezoidal(func, lower, upper, i)
            else:
                R[i,k] = R[i, k-1] + (R[i,k-1] - R[i-1, k-1]) / (2 ** (2*k) - 1)

    result = R[n-1,n-1]
    del R

    return result

if __name__ == "__main__":

    # problem 1 : interpolation with 4-point
    n_nodes = 4
    lower = 0
    upper = 2

    nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper, 'type_1')
    nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

    coeffs_cheb = compute_coeffs(func, nodes_cheb)
    coeffs_uniform = compute_coeffs(func, nodes_uniform)

    xs = generate_uniform_nodes(32, lower - 0.5, upper + 0.5)
 
    y_cheb = compute_polynomials(xs, coeffs_cheb, nodes_cheb)
    y_uniform = compute_polynomials(xs, coeffs_uniform, nodes_uniform)
    y_real = np.array([func(x) for x in xs])
    y_nodes_uniform = np.array([func(x) for x in nodes_uniform])
    y_nodes_cheb = np.array([func(x) for x in nodes_cheb])

    plt.figure(0, figsize = (8,6))
    plt.plot(xs, y_real, 'r-',label = 'sin(pi * x)')
    plt.plot(xs, y_cheb, 'b-',label = 'interpolate with chebshev')
    plt.plot(xs, y_uniform, 'k-',label = 'interpolate with uniform')
    plt.plot(nodes_cheb, y_nodes_cheb, 'bo', label = 'chebyshev nodes')
    plt.plot(nodes_uniform, y_nodes_uniform, 'ko', label = 'uniform nodes')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xlim([0,2.0])
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.title("Newton polynomial interpolation with 4 point")
    plt.savefig("./interpolation_4_points.png")

    # problem 1 : 5 - point interpolation
    n_nodes = 5
    nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper, 'type_1')
    nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

    coeffs_cheb = compute_coeffs(func, nodes_cheb)
    coeffs_uniform = compute_coeffs(func, nodes_uniform)

    y_cheb = compute_polynomials(xs, coeffs_cheb, nodes_cheb)
    y_uniform = compute_polynomials(xs, coeffs_uniform, nodes_uniform)
    y_nodes_uniform = np.array([func(x) for x in nodes_uniform])
    y_nodes_cheb = np.array([func(x) for x in nodes_cheb])

    plt.figure(1, figsize = (8,6))
    plt.plot(xs, y_real, 'r-',label = 'sin(pi * x)')
    plt.plot(xs, y_cheb, 'b-',label = 'interpolate with chebshev')
    plt.plot(xs, y_uniform, 'k-',label = 'interpolate with uniform')
    plt.plot(nodes_cheb, y_nodes_cheb, 'bo', label = 'chebyshev nodes')
    plt.plot(nodes_uniform, y_nodes_uniform, 'ko', label = 'uniform nodes')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xlim([0,2.0])
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.title("Newton polynomial interpolation with 5 point")
    plt.savefig("./interpolation_5_points.png")

    # use different n_nodes
    n_nodes_list = [4, 6, 8, 12]
    y_cheb_list = []
    y_uniform_list = []

    for n_nodes in n_nodes_list:
        nodes_cheb = generate_chebyshev_nodes(n_nodes, lower, upper, 'type_1')
        nodes_uniform = generate_uniform_nodes(n_nodes, lower, upper)

        coeffs_cheb = compute_coeffs(func, nodes_cheb)
        coeffs_uniform = compute_coeffs(func, nodes_uniform)

        y_cheb = compute_polynomials(xs, coeffs_cheb, nodes_cheb)
        y_uniform = compute_polynomials(xs, coeffs_uniform, nodes_uniform)

        y_cheb_list.append(y_cheb)
        y_uniform_list.append(y_uniform)

    plt.figure(2)
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (16,6))
    ax1.set_title("N-nodes interpolation with uniform grid")
    ax2.set_title("N-nodes interpolation with chebyshev grid")

    ax1.plot(xs, y_real, 'r-',label = 'sin(pi * x)')
    ax2.plot(xs, y_real, 'r-',label = 'sin(pi * x)')

    for n_nodes, y_cheb, y_uniform in zip(n_nodes_list, y_cheb_list, y_uniform_list):
        ax1.plot(xs, y_uniform, label = "n_nodes : {} uniform".format(n_nodes))
        ax2.plot(xs, y_cheb, label = "n_nodes : {} cheb".format(n_nodes))

    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("y-axis")
    ax1.set_xlim([0,2.0])   
    ax1.set_ylim([-1.5, 1.5]) 

    ax2.set_xlabel("x-axis")
    ax2.set_ylabel("y-axis")
    ax2.set_xlim([0,2.0])   
    ax2.set_ylim([-1.5, 1.5]) 

    ax1.legend()
    ax2.legend()

    plt.savefig("./interpolation_n_nodes.png")

    # problem 2
    true_value = 0 # integral sin(pi * x) dx = 0 for x in [0,2]

    # compute Newton polynomial for interpolation
    n_nodes_polynomial= 4
    nodes_polynomial = generate_uniform_nodes(n_nodes_polynomial, lower, upper)
    coeffs_polynomial = compute_coeffs(func, nodes_polynomial)
    func_poly = lambda x : compute_polynomial(x, coeffs_polynomial, nodes_polynomial)
    
    # 1-step : compute simpson integral with different nodes and compare the results
    # First, compute simpson integral with 8 nodes and compare the results between sin(pi * x) and newton polynomial
    simpson_integral = compute_simpson(func, lower, upper, n_nodes = 8)
    simpson_integral_poly = compute_simpson(func_poly, lower, upper, n_nodes = 8)

    print("(Simpson method) Real value : {:.8f}, sin(pi * x) : {:.8f}, polynomial : {:.8f}".format(
        true_value, simpson_integral, simpson_integral_poly
    ))

    # 2-step : compare the results with increasing the node number for the integration
    print("="*8, "Compare the results with respect to the node number for the integration", "=" * 8)

    n_nodes_list = [4, 8, 16, 32, 64, 128]
    err_list = []
    err_list_poly = []

    for n_nodes in n_nodes_list:
        simpson_integral = compute_simpson(func, lower, upper, n_nodes = n_nodes)
        simpson_integral_poly = compute_simpson(func_poly, lower, upper, n_nodes = n_nodes)

        print("(Simpson method) Real value : {:.2f}, sin(pi * x) : {:.8f}, polynomial : {:.8f}, n_nodes : {},".format(
            true_value, simpson_integral, simpson_integral_poly, n_nodes
        ))

        err_list.append(np.abs(true_value - simpson_integral))
        err_list_poly.append(np.abs(true_value - simpson_integral_poly))
    
    plt.figure(3, figsize = (8,6), clear = True)
    plt.plot(n_nodes_list, err_list, 'ro-', label = 'absolute error for sin(pi * x)')
    plt.plot(n_nodes_list, err_list_poly, 'bo-', label = 'absolute error for Newton Polynomial')
    plt.xlabel("Number of nodes for integration")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Comparision with increasing the node number (Simpson Algorithm)")
    plt.savefig("./comparision_integral.png")
    
    # problem 2 : use romberg algorithm to compute integral with different nodes
    print("="*8, "Another integral algorithm : Romberg Algorithm for high order accuracy", "=" * 8)
    n_romberg = 2
    romberg_integral = romberg_algorithm(func, lower, upper, n_romberg) # romberg algorithm for sin(pi*x)

    # compute romberg algorithm : integral of the polynomial interpolation
    romberg_integral_poly = romberg_algorithm(func_poly, lower, upper, n_romberg)

    print("(Romberg method) Real value : {:.2f}, sin(pi * x) : {:.8f}, polynomial : {:.8f}".format(
        true_value, romberg_integral, romberg_integral_poly
    ))