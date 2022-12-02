''' 
Homework03
- editor : JinSu KIM (2019-27420)
- List : see the description.png
'''
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.auto import tqdm
from typing import List, Callable, Union, Literal, Optional

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
        self.u = np.zeros((nx,ny))
        self.D = np.zeros((nx,ny))

class FEMsolver:
    def __init__(self, lx : float, ly : float, nx : int, ny : int):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.hx = lx / nx
        self.hy = ly / ny
        
    def mesh(self):
        pass

    def solve(self):
        pass

if __name__ == "__main__":
    pass