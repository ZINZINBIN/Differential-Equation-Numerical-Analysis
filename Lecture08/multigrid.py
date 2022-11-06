''' Multigrid methods
    List
    - Iteration method : Jacobi, Gauss-Seidel, SOR method
    Reference
    - https://medium.com/analytics-vidhya/applied-computational-thinking-using-python-multigrid-methods-64c86113e60b
    - OpenMG: A New Multigrid Implementation in Python
    - https://github.com/tsbertalan/openmg
    Example for applying Multigrid method
    - https://github.com/sanghvirajit19/Laplacian_Problem/blob/working/Jacobi_Laplacian_Problem%20(Ax%20%3D%20b).ipynb
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable, List, Literal, Union, Optional, Tuple
from tqdm.auto import tqdm

# smoother : iterative smoother which solve approximate solution of Ax = b
# Solver list
def JacobiSolve(A : np.ndarray, B : Union[np.ndarray, np.array], x : Union[np.ndarray, np.array], n_iters : int, threshold : Optional[float] = None, is_print : bool = True, boundary : Optional[Callable] = None):
    
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
        
        if boundary:
            boundary(x_next)
            x_next = x_next.reshape(-1,1)
    
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

    if is_converge and is_print:
        print("Gauss-Seidel iteration converged at iteration : {}, thres: {:.3f}".format(n_iter + 1, residual))

    return x_next.reshape(-1,)

# Multigrid method
# restrict : fine operator to coarse operator
def restriction(N : int, shape : Tuple[int,...], dense : bool = False):
    # N : total number of components which consist the matrix A
    # shape : the real world structure
    
    assert len(shape) <= 3, "dimension of array should be less than 3"
    dims = len(shape)
    
    if dense:
        R = np.zeros((int(N/(2**dims)), int(N)))
    else:
        R = sp.sparse.lil_matrix((int(N/(2**dims)), int(N)))
        
    r = 0
    
    if dims == 1:
        nx = shape[0]
    elif dims == 2:
        nx, ny = shape[0], shape[1]
    else:
        nx,ny,nz = shape[0], shape[1], shape[2]
    
    each = 1.0 / (2**dims)
    
    if dims == 1:
        coarse_arr = np.array(range(N)).reshape(shape)[::2].ravel()
    elif dims == 2:
        coarse_arr = np.array(range(N)).reshape(shape)[::2,::2].ravel()
    else:
        coarse_arr = np.array(range(N)).reshape(shape)[::2,::2,::2].ravel()
    
    for c in coarse_arr:
        
        R[r,c] = each
        R[r,c+1] = each
        
        if dims >= 2:
            R[r, c + nx] = each
            R[r, c + nx + 1] = each
        
            if dims == 3:
                R[r, c + nx * ny] = each
                R[r, c + nx * ny + 1] = each
                R[r, c + nx * ny + nx] = each
                R[r, c + nx * ny + nx + 1] = each
        
        r += 1
        
    return R

def restrictions(N : int, shape : Tuple[int,...], coarsest_level : int, dense : bool = False, verbose : bool = False):
    
    if verbose:
        print("Generating restriction matrices, dense={}".format(dense))
    
    dims = len(shape)
    levels = coarsest_level + 1
    
    R = list(range(levels - 1))
    
    for level in range(levels - 1):
        new_size = int(N / (2**(dims * level)))
        new_shape = tuple(np.array(shape) // (2**level))
        R[level] = restriction(new_size, new_shape, dense)
    
    return R

# coarsen array : Galerkin coarse-grid approximation, AH = R*A_h*R.T
def coarsen_A(A_in : np.ndarray, coarsest_level : int, R : List, dense : bool = False):
    levels = coarsest_level + 1
    A = list(range(levels))
    A[0] = A_in
    
    for level in range(1, levels):
        A[level] = np.matmul(np.matmul(R[level - 1], A[level - 1]), R[level - 1].T)

    return A

# smoother 
def smooth(A : np.ndarray, B : Union[np.ndarray, np.array], x : Union[np.ndarray, np.array], n_iters : int, threshold : Optional[float] = None, is_print : bool = True, solver : Literal['Jacobi','Gauss-Seidel'] = 'Gauss-Seidel'):
    if solver == 'Jacobi':
        return JacobiSolve(A,B,x,n_iters,threshold,is_print)
    else:
        return GaussSeidelSolve(A,B,x,n_iters,threshold,is_print)

# utility
def compute_residual(A:np.ndarray, B : Union[np.ndarray, np.array], x : Union[np.ndarray, np.array]):
    
    if type(x) == np.array:
        x = x.reshape(-1,1)
    
    if type(B) == np.array:
        B = B.reshape(-1,1)
    
    return B - np.matmul(A,x)

def compute_correction(R:np.ndarray, v : Union[np.ndarray, np.array]):
    v = v.reshape(-1,1)
    v_cor = np.matmul(R.T, v)
    return v_cor
    
# multigrid cycle : recursive formular
def v_cycle(
    x : np.ndarray,
    A : List, 
    B : np.ndarray, 
    R : List,
    n_iters : int, 
    threshold : Optional[float] = None,
    level : int = 0,
    coarsest_level : int = 4, 
    is_print : bool = True, 
    solver : Literal['Jacobi','Gauss-Seidel'] = 'Gauss-Seidel',
    dense : bool = False,
    verbose : bool = True
    ):
    
   
    A_level = A[level]
    R_level = R[level]
    B_level = B
    
    if level < coarsest_level - 1:
        x = smooth(A_level,B_level,x,n_iters,threshold,is_print,solver).reshape(-1,1)
        residual = compute_residual(A_level, B_level, x).reshape(-1,1)
        
        coarse_residual = np.matmul(R_level, residual).reshape(-1,1)
        B_level = np.matmul(R_level, B_level).reshape(-1,1)

        v = v_cycle(np.matmul(R_level, x).reshape(-1,1), A, coarse_residual, R, n_iters, threshold, level+1, coarsest_level, is_print, solver, dense, verbose)
        v_cor = compute_correction(R_level, v)
        x += v_cor
        
    else:
        x = smooth(A_level,B_level, x, n_iters, threshold, is_print, solver)
        
    return x
    
    
# generate laplace equation for real-physics problem
def boundary(f, n):
    
    # Boundary conditions
    # Top
    f[0, :(n // 4)] = np.arange(13, 5, -(13 - 5) / (n // 4))
    f[1, :(n // 4)] = np.arange(13, 5, -(13 - 5) / (n // 4))

    f[:2, (n // 4):(3 * n // 4)] = 5

    f[0, (3 * n // 4):] = np.arange(5, 13, (13 - 5) / (n // 4))
    f[1, (3 * n // 4):] = np.arange(5, 13, (13 - 5) / (n // 4))

    # Bottom
    f[n-2:, :] = 21

    # Left
    f[:(3 * n // 8), 0] = np.arange(13, 40, ((40 - 13) / (3 * n // 8)))
    f[:(3 * n // 8), 1] = np.arange(13, 40, ((40 - 13) / (3 * n // 8)))

    f[(n // 2):, 0] = np.arange(40, 21, -((40 - 21) / (n // 2)))
    f[(n // 2):, 1] = np.arange(40, 21, -((40 - 21) / (n // 2)))

    # Right
    f[:(n // 2), -1] = np.arange(13, 40, ((40 - 13) / (n // 2)))
    f[:(n // 2), -2] = np.arange(13, 40, ((40 - 13) / (n // 2)))

    f[(5 * n // 8):, -1] = np.arange(40, 21, -((40 - 21) / (3 * n // 8)))
    f[(5 * n // 8):, -2] = np.arange(40, 21, -((40 - 21) / (3 * n // 8)))

    # Heater
    f[(3 * n // 8):(n // 2) + 1, :(n // 8 + 1)] = 40

    f[(n // 2):(5 * n // 8) + 1, -(n // 8 + 1):] = 40
    
def Grid_with_BC(n):
    #grid
    f = np.zeros((n, n))

    boundary(f, n)

    return f  

# Initialization
def initgrid(n):
    #grid
    f = np.random.randn(n, n)

    boundary(f, n)

    return f

def diffusion_matrix(nx:int,ny:int, dx : float, dy : float):
    N = nx * ny
    A = np.zeros((N,N))
    
    # interior
    for idx_x in range(1,nx-1):
        for idx_y in range(1,ny-1):
            A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (2.0 / dx**2 + 2.0 / dy**2) * (-1)
            
            A[idx_x * ny + idx_y + 1, idx_x * ny + idx_y] = 1.0 / dy**2
            A[idx_x * ny + idx_y - 1, idx_x * ny + idx_y] = 1.0 / dy**2
            A[(idx_x - 1) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
            A[(idx_x + 1) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
            
    # boundary
    for idx_y in range(1,ny-1):
        idx_x = 0
        A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (-1.0 / dx**2 + 2.0 / dy**2) * (-1)
        A[idx_x * ny + idx_y + 1, idx_x * ny + idx_y] = 1.0 / dy**2
        A[idx_x * ny + idx_y - 1, idx_x * ny + idx_y] = 1.0 / dy**2
        
        A[(idx_x + 2)* ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
        A[(idx_x + 1)* ny + idx_y, idx_x * ny + idx_y] = 2.0 / dx**2 *(-1)
        
        idx_x = nx-1
        A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (-1.0 / dx**2 + 2.0 / dy**2) * (-1)
        A[idx_x * ny + idx_y + 1, idx_x * ny + idx_y] = 1.0 / dy**2
        A[idx_x * ny + idx_y - 1, idx_x * ny + idx_y] = 1.0 / dy**2
        
        A[(idx_x - 2)* ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
        A[(idx_x - 1)* ny + idx_y, idx_x * ny + idx_y] = 2.0 / dx**2 *(-1)
    
    for idx_x in range(1,nx-1):
        
        idx_y = 0
        A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (2.0 / dx**2 - 1.0 / dy**2) * (-1)
        A[(idx_x+1) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
        A[(idx_x-1) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
        
        A[idx_x* ny + idx_y + 2, idx_x * ny + idx_y] = 1.0 / dy**2
        A[idx_x* ny + idx_y + 1, idx_x * ny + idx_y] = 2.0 / dy**2 *(-1)
        
        idx_y = ny-1
        A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (2.0 / dx**2 - 1.0 / dy**2) * (-1)
        A[(idx_x+1) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
        A[(idx_x-1) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
        
        A[idx_x* ny + idx_y - 2, idx_x * ny + idx_y] = 1.0 / dy**2
        A[idx_x* ny + idx_y - 1, idx_x * ny + idx_y] = 2.0 / dy**2 *(-1)
        
    # corner
    idx_x = 0
    idx_y = 0
    
    A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (1.0 / dx**2 + 1.0 / dy**2)
    A[(idx_x+2) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
    A[(idx_x+1) * ny + idx_y, idx_x * ny + idx_y] = 2.0 / dx**2 * (-1)
    
    A[idx_x* ny + idx_y + 2, idx_x * ny + idx_y] = 1.0 / dy**2
    A[idx_x* ny + idx_y + 1, idx_x * ny + idx_y] = 2.0 / dy**2 *(-1)
    
    idx_x = nx-1
    idx_y = 0
    
    A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (1.0 / dx**2 + 1.0 / dy**2)
    A[(idx_x-2) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
    A[(idx_x-1) * ny + idx_y, idx_x * ny + idx_y] = 2.0 / dx**2 * (-1)
    
    A[idx_x* ny + idx_y + 2, idx_x * ny + idx_y] = 1.0 / dy**2
    A[idx_x* ny + idx_y + 1, idx_x * ny + idx_y] = 2.0 / dy**2 *(-1)
    
    idx_x = 0
    idx_y = ny-1
    
    A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (1.0 / dx**2 + 1.0 / dy**2)
    A[(idx_x+2) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
    A[(idx_x+1) * ny + idx_y, idx_x * ny + idx_y] = 2.0 / dx**2 * (-1)
    
    A[idx_x* ny + idx_y - 2, idx_x * ny + idx_y] = 1.0 / dy**2
    A[idx_x* ny + idx_y - 1, idx_x * ny + idx_y] = 2.0 / dy**2 *(-1)
    
    idx_x = nx-1
    idx_y = ny-1
    
    A[idx_x * ny + idx_y, idx_x * ny + idx_y] = (1.0 / dx**2 + 1.0 / dy**2)
    A[(idx_x-2) * ny + idx_y, idx_x * ny + idx_y] = 1.0 / dx**2
    A[(idx_x-1) * ny + idx_y, idx_x * ny + idx_y] = 2.0 / dx**2 * (-1)
    
    A[idx_x* ny + idx_y - 2, idx_x * ny + idx_y] = 1.0 / dy**2
    A[idx_x* ny + idx_y - 1, idx_x * ny + idx_y] = 2.0 / dy**2 *(-1)
    
    return A

if __name__ == "__main__":
    
    n_grid = 64
    x = np.linspace(-1.0, 1.0, n_grid)
    y = np.linspace(-1.0, 1.0, n_grid)
    
    dx = 1
    dy = 1
    
    A = Grid_with_BC(n_grid)
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(15, 5))

    # initial boundary condition
    X,Y = np.meshgrid(x,y, indexing='ij')
    pcr = axes[0].pcolor(Y,X,A)
    fig.colorbar(pcr, ax = axes[0], extend = 'max')
    axes[0].set_title("Initial condition")
    axes[0].set_xlabel("x(m)")
    axes[0].set_ylabel("y(m)")
    
    # solve as Jacobi and multigrid
    # Jacobi or Gauss-Seidel
    x_init = A.reshape(-1,1)
    b = np.zeros((n_grid*n_grid,1))
    D = diffusion_matrix(n_grid, n_grid, dx, dy)
    A_jacobi = JacobiSolve(D, b, x_init, n_iters = 128, threshold = 1e-8, is_print = True, boundary = lambda x : boundary(x.reshape(n_grid, n_grid),n_grid)).reshape(n_grid,n_grid)
    
    pcr = axes[1].pcolor(Y,X,A_jacobi)
    fig.colorbar(pcr, ax = axes[1], extend = 'max')
    axes[1].set_title("Jacobi method")
    axes[1].set_xlabel("x(m)")
    axes[1].set_ylabel("y(m)")
    
    # MultiGrid method
    A = Grid_with_BC(n_grid)
    x_init = A.reshape(-1,1)
    b = np.zeros((n_grid*n_grid,1))
    D = diffusion_matrix(n_grid, n_grid, dx, dy)
    
    R = restrictions(n_grid * n_grid, (n_grid, n_grid), coarsest_level=4, dense = True, verbose = True)
    D_multi = coarsen_A(D, coarsest_level=4, R = R, dense = True)
    
    A_multi = v_cycle(x_init, D_multi, b, R, 1, threshold=1e-8, level = 0, coarsest_level=4, is_print=True, solver = 'Jacobi', dense = True, verbose = True).reshape(n_grid, n_grid)
    
    pcr = axes[2].pcolor(Y,X,A_multi)
    axes[2].set_title("Multigrid method")
    axes[2].set_xlabel("x(m)")
    axes[2].set_ylabel("y(m)")
    fig.colorbar(pcr, ax = axes[2], extend = 'max')
    fig.tight_layout()
    plt.savefig("./multigrid_2D_Heat_diffusion.png")