import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal

def func_01(x : float):
    return np.cos(x) - x

def func_02(x : float):
    return np.exp(-x) - np.cos(x)

def func_03(x : float, a : float):
    return x**2 - a

def func_03_derivative(x : float):
    return 2 * x

def func_04(x : float):
    return np.sign(x - 2) * np.sqrt(x-2)

def func_04_derivative(x : float):
    return 1.0 / (2.0 * np.sqrt(x-2))

def func_05(x : float, m : int):
    return np.power(x - 1, m)

def func_05_derivative(x : float, m : int):
    return m * np.power(x-1, m-1)

def generate_uniform_nodes(n_nodes : int, lower : float, upper : float):
    result = np.array([i / n_nodes for i in range(n_nodes)])
    result = lower + (upper - lower) * result
    return result

def FixedPointIteration(func : Callable[[float],float], x0 : float, n_iters : int = 128, tol : float = 1e-6):
    xi = x0
    xf = x0
    x_sol = None
    is_success = False

    def _g_func(x : float):
        return func(x) + x

    for n_iter in range(n_iters):
        xf = _g_func(xi)

        # critiera
        err = np.abs(xf - xi)

        if err < tol:
            is_success = True
            x_sol = xf

        # update new points 
        xi = xf

    return x_sol, is_success

def NewtonMethod(func : Callable[[float],float], func_derivative : Callable[[float],float], x0 : float, n_iters : int = 128, tol : float = 1e-6):
    xi = x0
    xf = x0
    x_sol = None
    is_success = False

    def _g_func(x : float):
        return x - func(x) / func_derivative(x)

    for n_iter in range(n_iters):
        xf = _g_func(xi)

        # critiera
        err = np.abs(xf - xi)

        if err < tol:
            is_success = True

        # update new points 
        xi = xf

    x_sol = xf
    
    return x_sol, is_success


def SecantMethod(func : Callable[[float],float], x0 : float, x1 : float, n_iters : int = 128, tol : float = 1e-6):
    xi = x0
    xf = x1
    x_sol = None
    is_success = False
    
    def _fdm(func : Callable[[float], float], x_i : float, x_f : float):
        return (func(x_f) - func(x_i)) / (x_f - x_i)

    for n_iter in range(n_iters):
        
        # secant value
        s = _fdm(func, xi, xf)
        
        # update new points 
        xi = xf
        xf = xi - func(xi) / s

        # critiera
        err = np.abs(xf - xi)

        if err < tol:
            is_success = True
            
    x_sol = xf
    
    return x_sol, is_success

if __name__ == "__main__":

    x0 = 0.01
    n_iters = 1024
    tol = 1e-6

    x_sol, is_success = FixedPointIteration(func_01, x0, n_iters, tol)
    is_success = "true" if is_success else "false"

    print("(problem) x = cos(x), (solution) x = {:.5f}, converge : {}".format(x_sol, is_success))
    print("cos(x) : {:5f}, x : {:5f}".format(func_01(x_sol) + x_sol, x_sol)) 


    init_value_list = [np.pi / 4 * i for i in range(0,8)]

    for x_init in init_value_list:
        x_sol, is_success = FixedPointIteration(func_02, x_init, n_iters, tol)
        is_success = "true" if is_success else "false"

        print("(problem) exp(-x) = x, (init) x = {:.5f}, (solution) x = {:.5f}, converge : {}".format(x_init, x_sol, is_success))
    
    # visualization
    xs = generate_uniform_nodes(32, -5, 15)
    ys = np.array([func_02(x) for x in xs])

    plt.plot(xs, ys, label = 'exp(-x) - cos(x)')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("./plot.png")

    # Newton Method
    # example 01 : Find a square root of a natural number
    x_init = 1.0
    a = np.exp(1)

    x_sol, is_success = NewtonMethod(lambda x : func_03(x,a), func_03_derivative, x_init, n_iters, tol)
    print("(problem) sqrt(e), (init) x = {:.5f}, (solution) x = {:.5f}, (check) x^2 = {:.5f},  converge : {}".format(x_init, x_sol, x_sol**2, is_success))

    # example 02 : Find a solution of sgn(x-2) * sqrt(x-2) = 0
    x_init = 3.0
    x_sol, is_success = NewtonMethod(func_04, func_04_derivative, x_init, n_iters, tol)
    print("(problem) sgn(x-2) * sqrt(x-2), (init) x = {:.5f}, (solution) x = {:.5f}, converge : {}".format(x_init, x_sol, is_success))

    # example 03 : Find a solution of (x-1)^m = 0
    x_init = 3.0
    m = 3
    x_sol, is_success = NewtonMethod(lambda x : func_05(x, m), lambda x : func_05_derivative(x, m), x_init, n_iters, tol)
    print("(problem) (x-1)^m = 0, (init) x = {:.5f}, (solution) x = {:.5f}, converge : {}".format(x_init, x_sol, is_success))