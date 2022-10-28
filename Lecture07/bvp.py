import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Literal, Union

# These functions are for example : y"=y'+2y+cos(t)
def classical_RK4(func : Callable, yi : float, ti : float, dt : float):
    s1 = func(yi, ti)
    s2 = func(yi + s1 * dt/2, ti + dt / 2)
    s3 = func(yi + s2 * dt/2, ti + dt / 2)
    s4 = func(yi + s3*dt, ti + dt)
    yf = yi + (s1 + 2 * s2 + 2 * s3 + s4) * dt / 6
    return yf

def linear_shooting(func : Callable, xs : Union[np.array, List], yl : float, yr : float, yl_d1 : float, yl_d2 : float):
    lamda = 0
    xl = xs[0]
    xr = xs[-1]
    
    dx = xs[1] - xs[0]
    ys_1 = np.zeros_like(xs)
    ys_2 = np.zeros_like(xs)

    z1 = np.array([yl_d1, yl])
    z2 = np.array([yl_d2, yl])
    
    for i,x in enumerate(xs):
        ys_1[i] = z1[1]
        ys_2[i] = z2[1]
        
        z1 = classical_RK4(func, z1, x, dx)
        z2 = classical_RK4(func, z2, x, dx)
        
    # coefficient estimation
    lamda = (yr - ys_2[-1]) / (ys_1[-1] - ys_2[-1])
    
    ys = ys_1 * lamda + ys_2 * (1-lamda)
    return ys

def Non_linear_shooting(func : Callable, xs : Union[np.array, List], yl : float, yr : float, yl_d1 : float, yl_d2 : float, n_iters : int = 128, eps : float = 1e-6):
    
    def _forward_euler_method(func, xs, yi, yi_d):
        ys = []
        ys_d = []
        
        y = yi
        yd = yi_d
        dx = xs[1] - xs[0]
        
        for x in xs:  
            y_2d = func(x,y,yd)
            yd += dx * func(x,y,yd)
            y += yd * dx + 0.5 * y_2d * dx * dx
            
            ys.append(y)
            ys_d.append(yd)
            
        ys = np.array(ys)
        ys_d = np.array(ys_d)
        
        return ys, ys_d
    
    def g_func(x):
        ys, ys_d = _forward_euler_method(func, xs, yl, x)
        return ys[-1]
            
    # find initial 1st derivative y'(t = ti)
    xi = g_func(yl_d1)
    xf = g_func(yl_d2)
    
    is_converged = False
    
    for n_iter in range(n_iters):
        x = xi + (yr - g_func(xi)) * (xf - xi) / (g_func(xf) - g_func(xi))
        
        if abs(x - xf) < eps:
            is_converged = True
            break
        else:
            xi = xf
            xf = x
            
    if is_converged:
        print("# Non-linear shooting : secand method converged")
    else:
        print("# Non-linear shooting : secand method not converged")
        
    # explicit euler method
    ys, ys_d = _forward_euler_method(func, xs, yi, x)
    
    return ys, ys_d

# From Lecture 05 : Gaussian Elimination Algorithm
def Gaussian_Elimination_TriDiagonal(A_origin : np.ndarray, B_origin : Union[np.array, np.ndarray]):
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape[0], A.shape[1]
    assert m == n, 'matrix should be n x n shape'
    
    if type(B) == np.ndarray:
        B = B.reshape(-1,)
        
    X = np.zeros_like(B)
    
    # Forward Elimination
    for idx_j in range(1,n):
        A[idx_j, idx_j] = A[idx_j, idx_j] - A[idx_j, idx_j - 1] * A[idx_j-1, idx_j] / A[idx_j-1, idx_j-1]
        B[idx_j] = B[idx_j] - A[idx_j, idx_j - 1] * B[idx_j - 1] / A[idx_j-1, idx_j-1]
        A[idx_j, idx_j-1] = 0
        
    # Backward Substitution 
    X[-1] = B[-1] / A[-1,-1]
    for idx_i in [n - i - 1 for i in range(1,n)]:
        X[idx_i] = 1 / A[idx_i, idx_i] * (B[idx_i] - A[idx_i, idx_i +1] * X[idx_i + 1])

    return A,B,X

def uniform_grid(xl : float, xr : float, n : int):
    return np.linspace(xl, xr, num = n)

def FDM_1D_poisson(func : Callable, xs : Union[np.ndarray,List], h : float):
    T = len(xs)
    u = np.zeros_like(xs)
    f = np.array([func(x) * h * h for x in xs])
    
    A = np.zeros((T,T))
    for t in range(0,T):
        if t == 0:
            A[t,t] = -2
            A[t,t+1] = 1
        elif t == T-1:
            A[t,t-1] = 1
            A[t,t] = 2
        else:
            A[t,t-1] = 1
            A[t,t] = -2
            A[t,t+1] = 1
    
    _,_,u = Gaussian_Elimination_TriDiagonal(A,f)
    
    return u         

if __name__ == "__main__":
    
    # linear shooting examples
    def func_ls_example(z : np.array, t : float):
        zd = np.matmul(np.array([[1,2],[1,0]]),z) + np.array([1,0]) * np.cos(t)
        return zd
    
    n = 16; yi = -0.3; yf = -0.1
    xs = uniform_grid(0,np.pi / 2, n)
    
    # initial 1st derivative
    yi_d1 = 0.0
    yi_d2 = 1.0

    # linear shooting method
    ys = linear_shooting(func_ls_example, xs, yi, yf, yi_d1, yi_d2)
    ys_true = -(np.sin(xs) + 3 * np.cos(xs)) / 10
    
    # plot the line
    plt.figure(1)
    plt.plot(xs,ys, 'ro-', label = 'linear shooting')   
    plt.plot(xs,ys_true, label = 'y(t) = -(sin(t) + 3cos(t))/10')
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.savefig("./linear_shooting.png")
    
    # Non-linear shooting
    n = 32; yi = 0; yf = np.log(2)
    xs = uniform_grid(1.0, 2.0, n)
    ys_true = np.log(xs)
    
    def func_nls_example(x : float, y : float, yd : float):
        return (-1) * yd * yd - y + np.log(x)
    
    ys, ys_d = Non_linear_shooting(func_nls_example, xs, yi, yf, 0.0, 1.0, n_iters = 128, eps = 1e-6)
    plt.figure(2)
    plt.plot(xs,ys, 'ro-', label = 'Non-linear shooting')   
    plt.plot(xs,ys_true, label = 'y(t) = log(t)')
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.savefig("./non_linear_shooting.png")
    
    # FDM for 1D poisson equation
    # initiali value
    xi = -1; xf = 1; ui = 0; uf = 0; n = 16
    xs = uniform_grid(xi, xf, n)
    
    def func_FDM_1D(x : float):
        return 1 - x**2
    
    u = FDM_1D_poisson(func_FDM_1D, xs, (xf-xi) / n)
    
    plt.figure(3)
    plt.plot(xs,u, 'ro-', label = 'FDM : 1D poisson')   
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.savefig("./FDM_1D_poission.png")