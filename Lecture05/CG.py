# Conjugate Gradient method
# reference : https://en.wikipedia.org/wiki/Conjugate_gradient_method
import numpy as np
from typing import Callable, List, Literal, Union

def func():
    pass

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

if __name__ == "__main__":
    
    A = np.array([[4,-1,-1,0,0,0],[-1,4,0,-1,0,0],[-1,0,4,-1,-1,0],[0,-1,-1,4,0,-1],[0,0,-1,0,4,-1],[0,0,0,-1,-1,4]], dtype = np.float32)
    B = np.array([1,5,0,3,1,5], dtype = np.float32)
    x_0 = np.array([0.25, 1.25, 0, 0.75, 0.25, 1.25])
    
    n_iters = 1024
    eps = 1e-8
    
    x_cg = conjugate_gradient(A,B,x_0, n_iters, eps)
    
    print("x_cg : ", x_cg)    