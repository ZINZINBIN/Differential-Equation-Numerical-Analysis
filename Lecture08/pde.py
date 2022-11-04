import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal, Union
from tqdm.auto import tqdm

# Problem 1 : ut = 4uxx + 1, 0<x<1 and t>0
def parabolic_matrix_1D(dx : float, dt : float, Ns : int, Nt : int):
    
    r = 2 * dt / dx **2
    
    D = np.zeros((Ns,Ns))
    B = np.zeros((Ns,1))
    
    for idx_i in range(0,Ns):
        
        if idx_i != 0 and idx_i != Ns-1:
            D[idx_i,idx_i] = 1 - 2 * r
            D[idx_i,idx_i-1] = r
            D[idx_i,idx_i+1] = r
        elif idx_i == 0:
            D[idx_i,idx_i] = 1 + r
            D[idx_i,idx_i+1] = (-1) * 2 * r
            D[idx_i,idx_i+2] = r
        elif idx_i == Ns-1:
            D[idx_i,idx_i] = 1 + r
            D[idx_i,idx_i-1] = (-1) * 2 * r
            D[idx_i,idx_i-2] = r
            
        B[idx_i] = dt

    return D,B

def solve_parabolic_1d_pde(D : np.ndarray, B : np.ndarray, Ns : int, Nt : int):
    
    # initialize u(x,t)
    u = np.zeros((Ns,Nt))

    for idx_t in tqdm(range(0,Nt-1), desc = "solve parabolic 1d pde"):
        
        # boundary condition
        u[0,idx_t] = 0
        u[-1,idx_t] = 0

        u[:,idx_t+1] = (np.matmul(D, u[:,idx_t].reshape(-1,1)) + B).reshape(-1,)
        
    return u

if __name__ == "__main__":
    
    # Problem 1 : 
    Ns = 32
    xl = 0
    xr = 1
    dx = (xr - xl) / Ns
    dt = 0.025 * dx ** 2 # dt < 0.25 * dx ** 2 for stability
    
    tl = 0
    tr = 0.1
    Nt = int((tr-tl)/dt)
    
    # generate diffusion matrix
    D,B = parabolic_matrix_1D(dx, dt, Ns, Nt)
    
    print("D : ", D.shape)
    print("B : ", B.shape)
    
    # solve equation using FDM and euler forward method
    u = solve_parabolic_1d_pde(D,B,Ns,Nt)
    
    # plot the result
    x = np.linspace(xl, xr, Ns)
    t = np.linspace(tl, tr, Nt)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))

    axes[0].plot(x, u[:,0], label='Initial condition')
    axes[0].plot(x, u[:,int(0.5*Nt)], color='green', label="t={:.2f}".format(t[int(0.5*Nt)]))
    axes[0].plot(x, u[:,-1], color='brown', label="t={:.2f}".format(tr))
    
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].legend(loc = "upper right")
    
    X,T = np.meshgrid(x,t, indexing='ij')
    pcr = axes[1].pcolor(u)
    fig.colorbar(pcr, ax = axes[1], extend = 'max')
    
    fig.tight_layout()
    
    plt.savefig("./profile_1d.png")