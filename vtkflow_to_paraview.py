from vtk import vtkStructuredGridReader
from vtk.util import numpy_support as VN
from pathlib import Path
import numpy as np
import pandas as pd
import quickpiv as quickpiv



input_path = Path("Z:\\Room224_SharedFolder\\Alice\\pescoid\\20250316_153558_expzacy0047_fused\\done\\expzacy0047_pos1_piv_output_paraview")
voxel_dims = (0.406,0.406,2)
interSize = (40,40,8)
overlap=0



def write_vtk_txt(path,chunks,velocity):
    # Create the header lines
    header_lines = [
        "# vtk DataFile Version 3.0",
        "quickPIV vectors registered",
        "ASCII",
        "DATASET STRUCTURED_GRID",
        "DIMENSIONS 50 40 31",
        "POINTS 62000 float"
    ]

    additional_lines = [
        "POINT_DATA 62000",
        "VECTORS velocity float"
    ]


    with open(path, 'w') as f:

        for line in header_lines:
            f.write(line + '\n')

        np.savetxt(f, chunks, fmt='%.12f')

        for line in additional_lines:
            f.write(line + '\n')

        np.savetxt(f, velocity, fmt='%.12f')

    

def fullgrid_from_flow(input_path,voxel_dims,interSize,overlap):
    filenames = input_path.glob("*flow*")
    filenames = np.sort([f for f in filenames])
    for f in filenames:
        vtk_data = quickpiv.vtk_structured_grid(f)

        chunks = VN.vtk_to_numpy(vtk_data.GetPointData().GetArray('chunks'))
        velocity = VN.vtk_to_numpy(vtk_data.GetPointData().GetVectors())

        vector_grid = np.zeros(vtk_data.GetDimensions(),dtype=np.float64)
        vector_grid = np.tile(vector_grid,reps=(3,1,1,1))
        vector_grid.shape

        for i in range(len(chunks)):
            vector_grid[:,int(chunks[i][0]-1),int(chunks[i][1]-1),int(chunks[i][2]-1)] = velocity[i]

        full_velocity = np.stack((np.ravel(vector_grid[0]),np.ravel(vector_grid[1]),np.ravel(vector_grid[2])),axis=1)

        full_chunks = [idx for idx in np.ndindex(vtk_data.GetDimensions())]
        full_chunks = np.array(full_chunks)+1

        full_chunks = quickpiv.vector_origin(full_chunks,interSize, overlap)
        full_chunks = full_chunks * voxel_dims

        df = pd.DataFrame(np.concatenate((full_chunks,full_velocity),axis=1),columns=['x','y','z','vX','vY','vZ'])
        df.sort_values(by=['z','y','x'],axis=0,inplace=True)
        full_chunks = np.asarray(df.loc[:,['x','y','z']])
        full_velocity = np.asarray(df.loc[:,['vX','vY','vZ']])
        
        write_vtk_txt(input_path.joinpath(f.stem[:-3]+"paraview.vtk"),full_chunks,full_velocity)
        print(f"Saved full paraview file for {f.stem} ")

fullgrid_from_flow(input_path,voxel_dims,interSize,overlap)
