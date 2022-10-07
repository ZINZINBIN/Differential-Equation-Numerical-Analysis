import numpy as np
from typing import Union

def Naive_Gaussian_Elimination(A_origin : np.ndarray, B_origin : Union[np.array, np.ndarray]):
    
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape[0], A.shape[1]
    assert m == n, 'matrix should be n x n shape'
    
    if type(B) == np.ndarray:
        B = B.reshape(-1,)
        
    X = np.zeros_like(B)
    
    # Forward Elimination
    for idx_j in range(0,n-1):
        for idx_i in range(idx_j, n):
            A[idx_i,:] = A[idx_i, :] - A[idx_j, :] * A[idx_i, idx_j] / A[idx_j, idx_j]
            B[idx_i] = B[idx_i] - B[idx_j] * A[idx_i, idx_j] / A[idx_j, idx_j]
           
    # Backward Substitution 
    X[-1] = B[-1] / A[-1,-1]
    for idx_i in range(0,n-1,-1):
        X[idx_i] = B[idx_i] - np.dot(A[idx_i,:] * X)
        X[idx_i] /= A[idx_i, idx_i]
        
    return A,B,X

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

'''
- Jacobi iteration
- Gauss-Seidel iteration
- Successive Over Relaxation
'''

# numpy norm이 있으나, 우선 computational cost를 확인하기 위해 직접 norm 계산하는 것으로 대체
def norm(vec : np.array):
    n = len(vec)
    result = 0
    for i in range(n):
        result += vec[i] ** 2
    result /= n
    return np.sqrt(result)

def matmul_vec(A : np.ndarray, x : np.array):
    result = np.zeros_like(x)
    n = len(x)
    for idx_i in range(0, n):
        s = 0
        for idx_j in range(0,n):
            s += A[idx_i, idx_j] * x[idx_j]
        result[idx_i] = s
    return result

def Jacobi_Iteration(A_origin : np.ndarray, B_origin : Union[np.array, np.ndarray], x_0 : Union[np.array, np.ndarray], n_iters : int, eps : float):
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape[0], A.shape[1]
    assert m == n, 'matrix should be n x n shape'
    
    if type(B) == np.ndarray:
        B = B.reshape(-1,)
        
    if type(x_0) == np.ndarray:
        x_0 = x_0.reshape(-1,)
        
    x_prev = np.copy(x_0)
    x_next = np.copy(x_0)
    is_converged = False
    for n_iter in range(n_iters):
        for idx_i in range(0,n):
            s = 0
            for idx_j in range(0,n):
                if idx_i != idx_j:
                    s += A[idx_i, idx_j] * x_prev[idx_j]

            x_next[idx_i] = B[idx_i] - s
            x_next[idx_i] /= A[idx_i, idx_i]
    
        # check if converged
        residual = matmul_vec(A, x_next) - B
        
        if norm(residual) < eps:
            is_converged = True
            break
        else:
            x_prev = x_next
    
    if is_converged:
        print("Jacobi iteration converged at n_iters : {}".format(n_iter))
    else:
        print("Jacobi iteration not converged".format(n_iter))
    
    return x_next

def GaussSeidel_Iteration(A_origin : np.ndarray, B_origin : Union[np.array, np.ndarray], x_0 : Union[np.array, np.ndarray], n_iters : int, eps : float):
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape[0], A.shape[1]
    assert m == n, 'matrix should be n x n shape'
    
    if type(B) == np.ndarray:
        B = B.reshape(-1,)
        
    if type(x_0) == np.ndarray:
        x_0 = x_0.reshape(-1,)
        
    x = np.copy(x_0)
    is_converged = False
    
    for n_iter in range(n_iters):
        for idx_i in range(0,n):
            s = 0
            for idx_j in range(0,n):
                if idx_i != idx_j:
                    s += A[idx_i, idx_j] * x[idx_j]

            x[idx_i] = B[idx_i] - s
            x[idx_i] /= A[idx_i, idx_i]
    
        # check if converged
        residual = matmul_vec(A, x) - B
        
        if norm(residual) < eps:
            is_converged = True
            break
    
    if is_converged:
        print("Guass iteration converged at n_iters : {}".format(n_iter))
    else:
        print("Gauss iteration not converged".format(n_iter))
    
    return x

def SOR(A_origin : np.ndarray, B_origin : Union[np.array, np.ndarray], x_0 : Union[np.array, np.ndarray], n_iters : int, eps : float, w : float):
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape[0], A.shape[1]
    assert m == n, 'matrix should be n x n shape'
    
    if type(B) == np.ndarray:
        B = B.reshape(-1,)
        
    if type(x_0) == np.ndarray:
        x_0 = x_0.reshape(-1,)
        
    x = np.copy(x_0)
    is_converged = False
    
    for n_iter in range(n_iters):
        for idx_i in range(0,n):
            s = 0
            for idx_j in range(0,n):
                if idx_i != idx_j:
                    s += A[idx_i, idx_j] * x[idx_j]

            x[idx_i] = (1-w) * x[idx_i] + w * (B[idx_i] - s) / A[idx_i, idx_i]
    
        # check if converged
        residual = matmul_vec(A, x) - B
        
        if norm(residual) < eps:
            is_converged = True
            break
    
    if is_converged:
        print("SOR converged at n_iters : {}".format(n_iter))
    else:
        print("SOR not converged".format(n_iter))
    
    return x

    
if __name__ == "__main__":
    
    A = np.array([[20,5,0,0,0],[5,15,5,0,0],[0,5,15,5,0],[0,0,5,15,5],[0,0,0,5,10]], dtype = np.float32)
    B = np.array([1100, 100, 100, 100, 100], dtype = np.float32)
    
    # A_,B_,x = Naive_Gaussian_Elimination(A,B)
    A_,B_,x = Gaussian_Elimination_TriDiagonal(A,B)
    print("X : ", x)
    
    # Jacobi method
    n_iters = 1024
    eps = 1e-8
    
    A = np.array([[4,-1,-1,0,0,0],[-1,4,0,-1,0,0],[-1,0,4,-1,-1,0],[0,-1,-1,4,0,-1],[0,0,-1,0,4,-1],[0,0,0,-1,-1,4]], dtype = np.float32)
    B = np.array([1,5,0,3,1,5], dtype = np.float32)
    x_0 = np.array([0.25, 1.25, 0, 0.75, 0.25, 1.25])
    
    x_jacobi = Jacobi_Iteration(A,B,x_0,n_iters, eps)
    print("Jacobi : ", x_jacobi)
    print("Ax : ", np.matmul(A,x_jacobi))
    
    # Gause-Seidel method
    x_gauss = GaussSeidel_Iteration(A,B,x_0,n_iters, eps)
    print("Gauss : ", x_gauss)
    print("Ax : ", np.matmul(A,x_gauss))
    
    # SOR method
    w = 1.2
    x_sor = SOR(A,B,x_0,n_iters, eps,w)
    print("SOR : ", x_sor)
    print("Ax : ", np.matmul(A,x_sor))