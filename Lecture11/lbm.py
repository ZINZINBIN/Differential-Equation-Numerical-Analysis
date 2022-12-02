import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Union, Literal

if __name__ == '__main__':
    
    # random seed
    np.random.seed(42)
    
    # parameters
    dt = 1.0
    dx = 1.0
    tau = 0.75
    
    Nx = 128
    Nt = 128
    
    t0 = 0
    
    cs = dx / dt / np.sqrt(3)
    mu = cs * cs * (tau - 0.5 * dt) # diffusivity
    
    xs = np.array([i * dx for i in range(Nx)])
   
    # Lattice speeds and weights
    Nl = 2 # D1Q2
    w = np.array([0.5, 0.5])
    cx = np.array([1,-1])
    
    # initial condition
    rho = 1
    T0 = 1.0
    TL = 0
    L = 16
    
    # distribution
    f = np.ones((Nx, Nl)) * rho / Nl
    feq = np.zeros((Nx, Nl))
    
    # position and velocity
    x = np.zeros(Nx)
    u = np.zeros(Nx)
    
    # update boundary condition
    def update_boundary(f : np.ndarray, T0 : float, TL : float):
        # Dirichlet Boundary condition
        f[0,0] = T0 - f[0,1]
        f[-1,0] = TL - f[-1,1]
    
    # update distribution from Boltzmann euqation with BGK approixmation
    def update_distribution(f : np.ndarray, f0 : np.ndarray, dt : float, tau : float):
        f[:,0] += (f0[:,0] - f[:,0]) * dt / tau
        f[:,1] += (f0[:,1] - f[:,1]) * dt / tau
    
    # update equilibrium distribution from fluid variables
    def update_equilibrium(feq : np.ndarray, rho : np.array, w : np.array, vx : np.ndarray, ux : np.array, cs : float):
        feq[:,0] = rho * w[0] * (1 + vx[0] * ux / cs ** 2 + 0.5 * (vx[0] * ux) ** 2 / cs ** 4 + 0.5 * ux * ux / cs ** 2)
        feq[:,1] = rho * w[1] * (1 + vx[1] * ux / cs ** 2 + 0.5 * (vx[1] * ux) ** 2 / cs ** 4 + 0.5 * ux * ux / cs ** 2)
    
    plt.figure(figsize = (10,6))
    plt.xlabel("x-axis")
    plt.ylabel("Temperature (unit:K)")
    
    # initialize
    rho = np.array([1 for i in range(Nx)])
    ux = np.array([0 for i in range(Nx)])
    update_equilibrium(feq,rho,w,cx,ux,cs)
    
    print("f initial")
    print(np.dot(f, w.reshape(-1,1)).reshape(-1,))
    print("f equil")
    print(np.dot(feq, w.reshape(-1,1)).reshape(-1,))
    
    for idx_t in range(Nt): 
        t = t0 + idx_t * dt
        # compute fluid variables
        rho = np.sum(f, 1)
        ux = np.sum(np.matmul(f.reshape(Nx,Nl), cx.reshape(Nl,1)), 1) / rho
        
        # compute equilibrium distribution
        update_equilibrium(feq, rho, w, cx, ux, cs)
        
        # update distribution using Boltzmann equation
        update_distribution(f, feq, dt, tau)
        
        # update boundary condition
        update_boundary(f, T0, TL)
        
        if idx_t * 4 % Nt == 0:
            print("# t = {:.3f}".format(t))
            T = np.dot(f, w.reshape(-1,1)).reshape(-1,)
            plt.plot(xs, T, label = "t={:.3f}".format(t))

    plt.legend()
    plt.savefig("./lbm_1D_diffusion.png")