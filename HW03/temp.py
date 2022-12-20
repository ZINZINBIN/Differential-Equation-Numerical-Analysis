import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import dolfin as df

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


def load_mesh(mesh_file):

    mesh = df.Mesh()
    with df.XDMFFile(mesh_file) as infile:
        infile.read(mesh)
        # These are the markers
        ffun = df.MeshFunction("size_t", mesh, 2)
        infile.read(ffun, "name_to_read")
    return mesh, ffun


if __name__ == "__main__":
    msh_file = "tri_mesh_size_3.msh"
    mesh_file = "mesh.xdmf"
    line_file = "mf.xdmf"

    gmsh2dolfin(msh_file, mesh_file, line_file)
    mesh, ffun = load_mesh(mesh_file)
    
    V = df.FunctionSpace(mesh, 'CG', 2)
    
    domains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    
    # boundaries
    class BCline(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 0) < df.DOLFIN_EPS
        
    class FGline(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] + 5.0) < df.DOLFIN_EPS and abs(x[1] - 55) < 5.0
    
    boundary_bc = BCline()
    boundary_fg = FGline()
    
    boundaries = df.MeshFunction("size_t", mesh, 1, 0)
    boundary_bc.mark(boundaries, 1)
    boundary_fg.mark(boundaries, 2)
  
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
    plt.savefig('./problem01_sol.png') 
    