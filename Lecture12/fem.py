''' 2D poission equation solver by FEM method
    - strong form : ∇2u(x,y) = f(x,y), x on Ω
    - boundary condition : u(x,y)=uD(x,y), x on ∂Ω
'''
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.auto import tqdm
from typing import List, Callable, Union, Literal, Optional

# iterative Linear matrix solver
def GaussSeidelSolve(A : np.ndarray, B : Union[np.ndarray, np.array], x : Union[np.ndarray, np.array], n_iters : int, threshold : Optional[float] = None):
    if type(x) == np.array:
        x = x.reshape(-1,1)
    
    if type(B) == np.array:
        B = B.reshape(-1,1)
        
    D = A.diagonal()
    L = np.tril(A, k = -1)
    U = np.triu(A, k = 1)
    
    N = A.shape[0]
    
    D_inv = D ** (-1)
    D_inv = np.diag(D_inv)
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
            
    return x_next.reshape(-1,)

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
        # ∇2 u(i,j) = (u(i+1,j) + u(i-1,j) - 2*u(i,j))/hx^2 + (u(i,j+1) + u(i,j-1) - 2*u(i,j))/hy^2
        self.u = np.zeros((nx*ny,1))
        self.D = np.zeros((nx*ny,nx*ny))
        
        # inside the region
        for idx_i in range(1,ny-1):
            for idx_j in range(1,nx-1):
                idx = idx_i * nx + idx_j
                idx_u = (idx_i + 1) * nx + idx_j
                idx_d = (idx_i - 1) * nx + idx_j
                idx_l = idx_i * nx + idx_j - 1
                idx_r = idx_i * nx + idx_j + 1
                
                self.D[idx_u,idx] = 1
                self.D[idx_d,idx] = 1
                self.D[idx_l,idx] = 1
                self.D[idx_r,idx] = 1
                self.D[idx,idx] = -4
                
        # boundary
        idx_i = 0
        
        idx_i = ny - 1
        
        idx_j = 0
        
        idx_j = nx - 1
        
                        
    def mapping(self, idx_i : int, idx_j : int):
        idx = idx_i * self.nx + idx_j
        return self.u[idx]
    
    def solve(self):
        pass
    
    def boundary(self):
        pass

class FEMsolver:
    def __init__(self, lx : float, ly : float, nx : int, ny : int):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.hx = lx / nx
        self.hy = ly / ny
        
        self.u = np.zeros((nx * ny, 1))
        self.shape_fn = np.zeros(())
        
    # Linear rectangular method
    def mesh(self):
        pass
    
    # solve locally
    def solve(self):
        pass
    
    def postprocessing(self):
        pass
    
    # boundary condition
    def boundary(self):
        pass

if __name__ == "__main__":
    pass