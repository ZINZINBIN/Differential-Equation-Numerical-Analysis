''' 
Homework03
- editor : JinSu KIM (2019-27420)
- List : see the description.png
- Reference
    (1) 
'''
import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import dolfin as df
from typing import Optional

# convert .msh file to dolfin mesh file
def gmsh2dolfin(msh_file, mesh_file, line_file=None):
    
    msh = meshio.gmsh.read(msh_file)

    line_cells = []
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "line":
            if len(line_cells) == 0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])

    line_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                line_data = np.vstack(
                    [line_data, msh.cell_data_dict["gmsh:physical"][key]]
                )
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]

    triangle_mesh = meshio.Mesh(
        points=msh.points,
        cells={"triangle": triangle_cells},
        cell_data={"name_to_read": [triangle_data]},
    )
    line_mesh = meshio.Mesh(
        points=msh.points,
        cells=[("line", line_cells)],
        cell_data={"name_to_read": [line_data]},
    )
    meshio.write(mesh_file, triangle_mesh)
    meshio.xdmf.write(line_file, line_mesh)

# load mesh file as mesh instance
def load_mesh(mesh_file):

    mesh = df.Mesh()
    with df.XDMFFile(mesh_file) as infile:
        infile.read(mesh)
        # These are the markers
        ffun = df.MeshFunction("size_t", mesh, 2)
        infile.read(ffun, "name_to_read")
    return mesh, ffun

# boundaries for 2D poisson equation
class BCline(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0) < df.DOLFIN_EPS
    
class FGline(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] + 5.0) < df.DOLFIN_EPS and abs(x[1] - 55) < 5.0

# plot the result
def plot_and_save(mesh, u, save_file : str = "test.png"):
    
    if u is None:
        print("solution u not found..!")
        return
 
    if mesh is None:
        print("mesh not found..!")
        return
    
    n = mesh.num_vertices()
    d = mesh.geometry().dim()

    # Create the triangulation
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in df.cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    # Plot the mesh
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 5))
    axes[0].triplot(triangulation)
    zfaces = np.asarray([u(cell.midpoint()) for cell in df.cells(mesh)])
    pcr = axes[1].tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    fig.colorbar(pcr, ax = axes[1], extend = 'max')
    
    fig.tight_layout()
    plt.savefig(save_file) 
    

class Solver:
    def __init__(self, msh_file : str, mesh_file : str, line_file : str, save_file : str):
        self.msh_file = msh_file
        self.mesh_file = mesh_file
        self.line_file = line_file
        self.save_file = save_file
        
        self.mesh = None
        self.ffun = None
        self.u = None
        self.v = None
        
        self.setup()
    
    def setup(self):
        gmsh2dolfin(self.msh_file, self.mesh_file, self.line_file)
        mesh, ffun = load_mesh(self.mesh_file)
        
        self.mesh = mesh
        self.ffun = ffun
        
        self.boundary_bc = BCline()
        self.boundary_fg = FGline()
    
    def solve(self):
        
        if self.mesh is None:
            return

        V = df.FunctionSpace(self.mesh, 'CG', 2)
        
        domains = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        
        boundaries = df.MeshFunction("size_t", self.mesh, 1, 0)
        self.boundary_bc.mark(boundaries, 1)
        self.boundary_fg.mark(boundaries, 2)
    
        bcs = [df.DirichletBC(V, df.Constant(100.0), boundaries, 1), df.DirichletBC(V, df.Constant(0.0), boundaries, 2)]
        
        dx = df.Measure("dx")[domains]
        ds = df.Measure("ds")[boundaries]
        
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        
        f = df.Constant(0)
        g = df.Constant(0)
        
        a = df.inner(df.grad(u), df.grad(v)) * dx
        L = f*v*dx + g*v*ds(0) + g*v*ds(1) + g*v*ds(2)
        
        u = df.Function(V)

        df.solve(a == L, u, bcs)
        
        self.u = u
        self.v = v
        
        print("Process done..!")
    
    def plot(self):
        plot_and_save(self.mesh, self.u, self.save_file)
    
    def update_argument(self, msh_file : Optional[str] = None, mesh_file : Optional[str] = None, line_file : Optional[str] = None, save_file : Optional[str] = None):
        if mesh_file:
            self.mesh_file = mesh_file
        
        if msh_file:
            self.msh_file = msh_file
        
        if line_file:
            self.line_file = line_file
        
        if save_file:
            self.save_file = save_file
            
        self.setup()

if __name__ == "__main__":
    
    # size 1
    solver = Solver(msh_file = "tri_mesh_size_1.msh", mesh_file = "mesh_size_1.xdmf", line_file = "mf_size_1.xdmf", save_file="./sol_mesh_size_1.png")
    solver.solve()
    solver.plot()
    
    # size 2
    solver.update_argument(msh_file = "tri_mesh_size_2.msh", mesh_file = "mesh_size_2.xdmf", line_file = "mf_size_2.xdmf", save_file="./sol_mesh_size_2.png")
    solver.solve()
    solver.plot()
    
    # size 3
    solver.update_argument(msh_file = "tri_mesh_size_3.msh", mesh_file = "mesh_size_3.xdmf", line_file = "mf_size_3.xdmf", save_file="./sol_mesh_size_3.png")
    solver.solve()
    solver.plot()