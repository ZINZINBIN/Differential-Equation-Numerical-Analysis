import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal, Union

def single_step_explicit(func : Callable, y_init : np.ndarray, h : float):
    y = y_init
    pass

def multi_step_explicit():
    pass

def ODE_func(A : np.ndarray, y : np.ndarray):
    y_diff = np.matmul(A,y)
    return y_diff

def Adams_Bashforth_2(func : Callable, h : float):
    # explict Adams-Bashforth method for order 2
    pass

if __name__ == "__main__":
   
    A = np.array([[-1.0, -0.75],[1.0, 0]])
    y_init = np.array([-2.5, 3.0])
    
    pass