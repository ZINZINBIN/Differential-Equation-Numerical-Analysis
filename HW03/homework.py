''' 
Homework03
- editor : JinSu KIM (2019-27420)
- List : see the description.png
- Reference
    (1) 
'''
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.auto import tqdm
from typing import List, Callable, Union, Literal, Optional

# Matrix solver : Conjugate Gradient Method for solving Ax = B
def conjugate_gradient(A_origin : np.ndarray, B_origin : np.ndarray, x_init : np.ndarray, n_iters : int, eps : float):
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape
    
    assert m == n, 'n_row and n_col should be same'
    
    def _compute_gradient(A : np.ndarray, B : np.ndarray, x : np.ndarray):
        return np.matmul(A,x) - B
    
    def _compute_residual(A : np.ndarray, B : np.ndarray, x : np.ndarray):
        return B - np.matmul(A,x)
    
    def _check_convergence(A : np.ndarray, B : np.ndarray, x : np.ndarray, eps : float):
        if np.sqrt(np.linalg.norm(np.matmul(A,x) - B)) < eps:
            return True
        else:
            return False
    
    x = np.copy(x_init)

    rk = _compute_residual(A,B,x)
    dk = _compute_gradient(A,B,x)
    
    is_converge = False
    
    for n_iter in range(n_iters):
        # Line search for step size a(k)
        ak = np.matmul(dk, rk) / np.matmul(dk, np.matmul(A,dk))
        
        # update x(k) = x(k-1) + a(k) * d(k)
        x = x + dk * ak
        
        # update residual r(k)
        rk = _compute_residual(A,B,x)
        
        # update gradient
        dk = _compute_gradient(A,B,x)
        
        is_converge = _check_convergence(A,B,x,eps)
        
        if is_converge:
            break
    
    if is_converge:
        print("Conjugate Gradient Method : converge at n_iter : {}".format(n_iter))
    else:
        print("Conjugate Gradient Method : not converged")
    return x

# for comparision, use FDM 
class FDMsolver:
    def __init__(self, lx : float, ly : float, nx : int, ny : int):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.hx = lx / nx
        self.hy = ly / ny
        
        # Discretization
        self._discretization(nx,ny,self.hx,self.hy)

    def _discretization(self, nx : int, ny : int, hx : float, hy :float):
        self.u = np.zeros((nx,ny))
        self.D = np.zeros((nx,ny))

class FEMsolver:
    def __init__(self, lx : float, ly : float, nx : int, ny : int):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.hx = lx / nx
        self.hy = ly / ny
        
    def mesh(self):
        pass

    def solve(self):
        pass

if __name__ == "__main__":
    A = np.random.randn(128,128)
    B = np.random.randn(128,1)
    x_init = np.zeros_like(B)
    
    x = conjugate_gradient(A, B, x_init, n_iters = 129, eps = 1e-6)