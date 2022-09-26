import numpy as np
import scipy as sp
from collections.abc import Callable

def func(x : float):
    return np.exp(x)

def forward_diff(func : Callable[[float],float], x : float, h : float = 1e-3):
    xr = func(x + h)
    xl = func(x)
    return (xr - xl) / h

def backward_diff(func : Callable[[float],float], x : float, h : float = 1e-3):
    xr = func(x)
    xl = func(x-h)
    return (xr - xl) / h

def central_diff(func : Callable[[float],float], x : float, h : float = 1e-3):
    xr = func(x + h)
    xl = func(x - h)
    return (xr - xl) / 2 / h

def integrate_trapezoid(func : Callable[[float],float], n_nodes : int, lower : float, upper : float):
    h = (upper - lower) / n_nodes
    xs = [lower + (upper - lower) / n_nodes * idx for idx in range(n_nodes)]
    ys = [func(x) for x in xs]
    result = sum(ys) * 2 - ys[0] - ys[-1]
    result *= h * 0.5
    return result

if __name__ == "__main__":
    # Numerical Differential
    h_lst = [0.1, 0.01]
    for h in h_lst:
        rd = func(0)
        fd = forward_diff(func, 0, h)
        bd = backward_diff(func, 0, h)
        cd = central_diff(func, 0, h)
        print("h : {}, fd : {:.5f}, bd : {:.5f}, cd : {:.5f}".format(h,np.abs(rd-fd),np.abs(rd - bd),np.abs(rd - cd)))

    # Numerical Integration
    # Trapezoidal
    true_value = func(2) - func(0)
    n_nodes_list = [n for n in range(5,320,8)]
    err_list = []

    for n_nodes in n_nodes_list:
        estimate_value = integrate_trapezoid(func, n_nodes, 0, 2)
        err = np.abs(estimate_value - true_value)
        err_list.append(err)

    err_list = np.array(err_list)
    n_nodes_optimal = n_nodes_list[np.argmax(err_list * (-1))]
    estimate_value_optimal = integrate_trapezoid(func, n_nodes_optimal, 0, 2)

    print("n_nodes : {}, estimate : {}, true : {}, err : {}".format(n_nodes_optimal, estimate_value_optimal, true_value, np.abs(estimate_value_optimal - true_value)))


    # Lagrange Polynomial
    def func(x : float):
        return 1 / (x + 1) ** 0.5

    result = 2 * np.sqrt(1)

    approx = np.sqrt(1/2) * (5/9 * func(-np.sqrt(3/5)) + 5/9 * func(np.sqrt(3/5)) + 8/9 * func(0))
    print("result : {}, approx : {}".format(result, approx))

    approx = np.sqrt(1/2) * (
        func(np.sqrt((5 - np.sqrt(40 / 7))/9)) * (322 + 13 * np.sqrt(70)) / 900 + 
        func(-np.sqrt((5 - np.sqrt(40 / 7))/9)) * (322 + 13 * np.sqrt(70)) / 900 + 
        func(np.sqrt((5 + np.sqrt(40 / 7))/9)) * (322 - 13 * np.sqrt(70)) / 900 + 
        func(-np.sqrt((5 + np.sqrt(40 / 7))/9)) * (322 - 13 * np.sqrt(70)) / 900 + 
        func(0) * 128 / 225
    )

    print("result : {}, approx : {}".format(result, approx))

