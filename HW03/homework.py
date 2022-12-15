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
    
    # f(x) = (Ax-b)^2 -> gradient f(x) = (Ax-b)
    def _compute_gradient(A : np.ndarray, B : np.ndarray, x : np.ndarray):
        return np.matmul(A,x) - B
    
    # residual : B - Ax
    def _compute_residual(A : np.ndarray, B : np.ndarray, x : np.ndarray):
        return B - np.matmul(A,x)
    
    def _check_convergence(A : np.ndarray, B : np.ndarray, x : np.ndarray, eps : float):
        res = np.sqrt(np.linalg.norm(np.matmul(A,x) - B))
        if res < eps:
            return True, res
        else:
            return False, res
    
    x = np.copy(x_init)

    rk = _compute_residual(A,B,x)
    dk = _compute_gradient(A,B,x) * (-1)
    
    is_converge = False
    
    for n_iter in range(n_iters):
        # Line search for step size a(k)
        ak = np.matmul(dk.T, rk) / np.matmul(dk.T, np.matmul(A,dk))
        
        # update x(k) = x(k-1) + a(k) * d(k)
        x = x + dk * ak
        
        # update residual r(k)
        rk = _compute_residual(A,B,x)
        
        # update gradient
        dk = _compute_gradient(A,B,x) * (-1)
        
        is_converge, residual = _check_convergence(A,B,x,eps)
        
        if is_converge:
            break
    
    if is_converge:
        print("# CGM | iteration: {:4d} | res: {:.5f} | converged".format(n_iter, residual))
    else:
        print("# CGM | iteration: {:4d} | res: {:.5f} | not converged".format(n_iter, residual))
    return x

# test function for debug
def test():
    pass

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
    
    A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    B = np.array([3,4,5,6]).reshape(-1,1)
    x_init = np.random.randn(4,1)
    
    x = conjugate_gradient(A, B, x_init, n_iters = 128, eps = 1e-6)