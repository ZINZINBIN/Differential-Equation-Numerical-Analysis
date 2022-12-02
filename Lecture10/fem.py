import numpy as np
from typing import Callable, List, Union, Literal

if __name__ =="__main__":
    A = np.array([[200,-50,0],[-50,150,-100],[0,-100,100]])
    B = np.array([0,0,10])
    A_inv = np.linalg.inv(A)
    
    x = np.dot(A_inv,B)
    print(x)
    print(np.matmul(A,x))