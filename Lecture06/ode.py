import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal, Union

def uniform_grid(xl : float, xr : float, n : int):
    return np.linspace(xl, xr, num = n)

def single_step_explicit(func : Callable, y_init : np.ndarray, h : float):
    pass

def multi_step_explicit():
    pass

def ODE_func(A : np.ndarray, y : np.ndarray):
    y_diff = np.matmul(A,y)
    return y_diff

def AB_2nd(func : Callable, ts : Union[np.array, List], z0 : Union[np.array, float]):
    # explicit Adams-Bashforth method for 2nd order
    ys = np.zeros_like(ts)
    ys_d = np.zeros_like(ts)
    
    dt = ts[1] - ts[0]
    
    z = z0
    z_prev = z0
    
    for i, t in enumerate(ts):
        
        ys[i] = z[1]
        ys_d[i] = z[0]
        
        if i == 0:
            s0 = func(z)
            z += dt * s0
        else:
            s0 = func(z)
            s1 = func(z_prev)
            z_prev = z
            z += dt * (3 * s0 - s1) * 0.5   
        
    return ys, ys_d

def ABM_2nd(func : Callable, ts : Union[np.array, List], z0 : Union[np.array, float]):
    # explicit Adams-Bashforth-Moulton methods for 2nd order
    ys = np.zeros_like(ts)
    ys_d = np.zeros_like(ts)
    
    dt = ts[1] - ts[0]
    
    z = z0
    z_next = z0
    
    for i, t in enumerate(ts):
        ys[i] = z[1]
        ys_d[i] = z[0]
        
        if i == len(ts)-1:
            s0 = func(z)
            z += dt * s0
        else:
            s0 = func(z)
            z_next = z + dt * s0
            s1 = func(z_next)
            z += dt * (s0 + s1) * 0.5   
        
    return ys, ys_d

if __name__ == "__main__":
   
    def exact_func(t : Union[float, np.array]):
        return 2 * np.exp(-0.5*t) + np.exp(-1.5*t)
    
    def exact_func_1d(t : Union[float, np.array]):
        return (-1)*np.exp(-0.5*t) - 1.5*np.exp(-1.5*t)
    
    def func(z : np.array):
        z_next = np.matmul(np.array([[-2, -0.75],[1, 0]]), z)
        return z_next
    
    ts = uniform_grid(0, 5.0, n = 32)
    ys_true = exact_func(ts)
    ys_1d_true = exact_func_1d(ts)
    
    z0 = np.array([-2.5, 3.0])
    # Adam-Bashforth 2nd order for explicit method
    ys_ab, ys_1d_ab = AB_2nd(func, ts, z0)
    
    z0 = np.array([-2.5, 3.0])
    # Adam-Bashforth-Moulton 2nd order for implicit method
    ys_abm, ys_1d_abm = ABM_2nd(func, ts, z0)
    
    plt.figure(1)
    plt.plot(ts, ys_ab, 'ro-', label = 'Adam-Bashforth 2nd order')
    plt.plot(ts, ys_abm, 'k*-', label = 'Adam-Bashforth-Moulton 2nd order')
    plt.plot(ts, ys_true, label = 'exact func')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("./Multistep_method.png")