''' 
Homework02
- editor : JinSu KIM (2019-27420)
- list
(1)  Consider a PDE u,tt = u,xx - (1+x2)u (0≤x≤1) with u(0,x) = 0, u(t,x) = u(t+2π,x), u(t,0) = sin(t), ∂xu(t,1) = 0. 
First, apply separation of variables. Next, describe how to solve the space-part equation, numerically.

(2) Make a FDM code for this PDE. u,t + m·x·u,x = 0 (m = 1 or -1) 
    Initial condition: u(0,x) = exp[-(x-x0)2]
    Domain: -L≤x≤L (You can choose L, but L >> |x0|)
    Boundary conditions: ∂xu(0,-L) = ∂xu(0,L) = 0
Draw or plot the state after some time. Discuss dependence on m and x0.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.auto import tqdm
from typing import List, Callable, Union, Literal, Optional

def FDM_prob02(nx : int, xl : float, xr : float, dt : float, m : int):
    dx = (xr - xl) / nx
    xs = np.linspace(xl, xr, nx)
    A = np.zeros((nx,nx))
    
    for idx_x in range(1,nx-1):
        A[idx_x,idx_x] = 1
        A[idx_x,idx_x+1] = xs[idx_x] * m * dt / 2 / dx
        A[idx_x,idx_x-1] = (-1) * xs[idx_x] * m * dt / 2 / dx
    
    idx_x = 0
    A[idx_x, idx_x] = 1 - 3 * xs[idx_x] * m * dt / 2 / dx
    A[idx_x, idx_x+1] = 4 * xs[idx_x] * m * dt / 2 / dx
    A[idx_x, idx_x+2] = (-1) * xs[idx_x] * m * dt / 2 / dx
        
    idx_x = len(xs) - 1
    A[idx_x, idx_x] = 1 + xs[idx_x] * m * dt / 2 / dx
    A[idx_x, idx_x-1] = (-4) * xs[idx_x] * m * dt / 2 / dx
    A[idx_x, idx_x-2] = 3 * xs[idx_x] * m * dt / 2 / dx
    
    return A

class PDESolver:
    def __init__(
        self, 
        tmax : float, 
        xmin : float, 
        xmax : float, 
        dx : float, 
        dt : float, 
        Boundary : Callable, 
        Initialize : Callable, 
        A : np.ndarray,
        save_dir : str,
        title : Optional[str] = None,
        plot_time_index : Optional[List] = [0,-1]
        ):
        # parameter update
        self.tmax = tmax
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.dt = dt
        self.ts = np.linspace(0, tmax, int(tmax/dt))
        self.xs = np.linspace(xmin,xmax, int((xmax-xmin)/dx))
        self.u = np.zeros((len(self.xs), len(self.ts)))
        self.A = A
        
        # Functions for use
        self.Boundary = Boundary
        self.Initialize = Initialize
        
        # parameters for plot
        self.save_dir = save_dir
        self.title = title
        self.plot_time_index = plot_time_index
        
    def solve(self):
        
        # initial condition
        self.Initialize(self.u)
        
        for idx_t in tqdm(range(0,len(self.ts)-1)):
            
            # Boundary condition
            self.Boundary(self.u)
            
            # update u(x,t+dt)
            u = self.u[:,idx_t].reshape(-1,1)
            self.u[:,idx_t+1] = np.matmul(self.A, u).reshape(-1,)
            
    def plot(self):
        
        plt.figure(figsize = (8,6), clear = True)
        
        for idx_t in self.plot_time_index:
            label = "t={:.3f}".format(self.ts[idx_t])
            plt.plot(self.xs, self.u[:,idx_t].reshape(-1,1), label = label)
        
        plt.xlabel("x-axis")
        plt.ylabel("u(x,t)")
        plt.legend()
        plt.title(self.title)
        plt.savefig(self.save_dir)

if __name__ == "__main__":
    
    # problem 02 : m = 1 case
    m = 1; L = 10; nx = 256; xl = (-1) * L; xr = L; dt = 1e-3; x0 = 1e-2
    A = FDM_prob02(nx, xl, xr, dt, m)
    
    dx = (xr - xl) / nx; tmax = 1.0
    
    ts = np.linspace(0, tmax, int(tmax/dt))
    xs = np.linspace(xl,xr,nx)
    
    def Initialize_prob02(u:np.ndarray, x : np.array, x0 : float):
        for idx_x in range(len(x)):
            u[idx_x,:] = np.exp(-(x[idx_x]-x0)**2)
    
    def Boundary_prob02(u:np.ndarray, x : np.array):
        idx_x = 0
        u[idx_x,0] = 3 * u[idx_x + 1, 0] - 2 * u[idx_x + 2, 0]
        
        idx_x = len(x) - 1
        u[idx_x,0] = 3 * u[idx_x - 1, 0] - 2 * u[idx_x - 2, 0]
        
    solver = PDESolver(
        tmax,
        xl,
        xr,
        dx,
        dt,
        lambda x : Boundary_prob02(x, xs),
        lambda x : Initialize_prob02(x, xs, x0),
        A,
        save_dir = "./problem02.png",
        title = "u(x,t) graph with different t",
        plot_time_index=[0,int(len(ts)/2), len(ts) -1]
    )
    
    solver.solve()
    
    solver.plot()