# Conjugate Gradient method
# reference : https://en.wikipedia.org/wiki/Conjugate_gradient_method
import numpy as np
from typing import Callable, List, Literal, Union

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

if __name__ == "__main__":
    
    A = np.array([[4,-1,-1,0,0,0],[-1,4,0,-1,0,0],[-1,0,4,-1,-1,0],[0,-1,-1,4,0,-1],[0,0,-1,0,4,-1],[0,0,0,-1,-1,4]], dtype = np.float32)
    B = np.array([1,5,0,3,1,5], dtype = np.float32)
    x_0 = np.array([0.25, 1.25, 0, 0.75, 0.25, 1.25])
    
    n_iters = 1024
    eps = 1e-8
    
    x_cg = conjugate_gradient(A,B,x_0, n_iters, eps)
    
    print("x_cg : ", x_cg)    