''' Multigrid methods
    List
    - Iteration method : Jacobi, Gauss-Seidel, SOR method
    Reference
    - https://medium.com/analytics-vidhya/applied-computational-thinking-using-python-multigrid-methods-64c86113e60b
    - OpenMG: A New Multigrid Implementation in Python
    - https://github.com/tsbertalan/openmg
'''
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal, Union, Optional
from tqdm.auto import tqdm

# Solver list
def JacobiSolve(A : np.ndarray, B : Union[np.ndarray, np.array], x : Union[np.ndarray, np.array], n_iters : int, threshold : Optional[float] = None, is_print : bool = True):
    
    if type(x) == np.array:
        x = x.reshape(-1,1)
    
    if type(B) == np.array:
        B = B.reshape(-1,1)
        
    D = np.tril(A, k = 0)
    L = np.tril(A, k = -1)
    U = np.tril(A, k = 1)
    N = A.shape[0]
    
    D_inv = D ** (-1)
    is_converge = False
    
    for n_iter in range(n_iters):
        
        dx = np.matmul((L+U), x)
        x_next = np.matmul(D_inv, B - dx)
    
        residual = np.linalg.norm(np.matmul(A,x_next) - B) / N
        residual = np.sqrt(residual)
        
        if residual < threshold:
            is_converge = True
            break
        else:
            x = x_next

    if is_converge and is_print:
        print("Jacobi iteration converged at iteration : {}, thres: {:.3f}".format(n_iter + 1, residual))

    return x_next.reshape(-1,)

def GaussSeidelSolve(A : np.ndarray, B : Union[np.ndarray, np.array], x : Union[np.ndarray, np.array], n_iters : int, threshold : Optional[float] = None, is_print : bool = True):
    
    if type(x) == np.array:
        x = x.reshape(-1,1)
    
    if type(B) == np.array:
        B = B.reshape(-1,1)
        
    D = np.tril(A, k = 0)
    L = np.tril(A, k = -1)
    U = np.tril(A, k = 1)
    N = A.shape[0]
    
    D_inv = D ** (-1)
    is_converge = False
    
    for n_iter in range(n_iters):
        
        dx = np.matmul((L+U), x)
        x_next = np.matmul(D_inv, B - dx)
    
        residual = np.linalg.norm(np.matmul(A,x_next) - B) / N
        residual = np.sqrt(residual)
        
        if residual < threshold:
            is_converge = True
            break
        else:
            x = x_next

    if is_converge and is_print:
        print("Gauss-Seidel iteration converged at iteration : {}, thres: {:.3f}".format(n_iter + 1, residual))

    return x_next.reshape(-1,)

# Multigrid method


if __name__ == "__main__":
    pass