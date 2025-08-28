'version: 1.0.0'
import numpy as np
from vtk import vtkStructuredGridReader,vtkStructuredGridWriter,vtkStructuredGrid,vtkPoints
from vtk.util import numpy_support as VN
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from glob import glob
import napari
import tifffile as tiff
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interpn
from pathlib import Path

#to do:
# get metadata from xml
# remove zero vectors
# scale to pixel dims

#read viventis metadata to get pixel dimensions
def read_metadata(input_path):

    metadata_path = glob(('\\').join(input_path.split('\\')[:-1])+'\\*companion*')
    with open(metadata_path[0],"r") as file:
        
        metadata = file.read()

    root = ET.fromstring(metadata)

    return root[0][0].attrib

#
def lazy_loading(input_path):
    # code adapted from https://napari.org/stable/tutorials/processing/dask.html
    filenames = sorted(glob(input_path), key=alphanumeric_key)
    # read the first file to get the shape and dtype
    # ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
    #sample = tiff.imread(filenames[0])

    # reads from ome meta_data file instead
    metadata = read_metadata(input_path)

    lazy_imread = delayed(tiff.imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn,key = range(0,int(metadata['SizeZ']))) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=(int(metadata['SizeZ']),int(metadata['SizeY']),int(metadata['SizeX'])), dtype=metadata['Type'])
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
    #stack.shape  # (nfiles, nz, ny, nx)

    # in jupyter notebook the repr of a dask stack provides a useful visual:
    
    return stack


#%% load a vtk file as a structured grid
def vtk_structured_grid(filename):
    reader = vtkStructuredGridReader()

    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    
    return reader.GetOutput()

#%% extract PIV window/chunk ID and associated vector velocity
def get_vtk_vectors(filename):
    #read data as a vtk structured grid
    data = vtk_structured_grid(filename)

    
    vectors = VN.vtk_to_numpy(data.GetPointData().GetVectors())
    
    chunks = VN.vtk_to_numpy(data.GetPoints().GetData())

    #vector_grid = np.zeros(data.GetDimensions(),dtype=np.float64)
    
    return chunks,vectors

#%% convert PIV chunk ID to center of each PIV window in image scale in pixels
def vector_origin(chunks,interSize,overlap):

    x_origins = (chunks[:,0] * interSize[0]) - (interSize[0]/2) - (overlap * chunks[:,0])
    y_origins = (chunks[:,1] * interSize[1]) - (interSize[1]/2) - (overlap * chunks[:,1])
    z_origins = (chunks[:,2] * interSize[2]) - (interSize[2]/2) - (overlap * chunks[:,2])

    return np.stack((x_origins,y_origins,z_origins),axis=1)

                
#%% combine vector origin with velocity and a given time step
def combine_vectors(vector_start,velocities,filename,scale_pixels=None):
    
    #create empty array each vector is a row and then there are two columns for the t,z,y,x values for both vector origin and velocity 
    total_vectors = np.zeros((len(vector_start),2,4))

    #extract time step from filename suffix
    time_point = int(filename.stem.split('_')[0][1:]) -1 #time_point = int(filename.split('\\')[-1].split('_')[0][1:]) -1
    
    #loop over all vectors and add them into the array
    for i in range(len(vector_start)):
        total_vectors[i,0,0] = time_point
        total_vectors[i,0,1:] = vector_start[i]

        total_vectors[i,1,0] = time_point
        total_vectors[i,1,1:] = velocities[i]
    
    #optional adjustment to convert values to image scale
    if scale_pixels == None:
        return total_vectors
    elif len(scale_pixels) == 3:
        return total_vectors * (1,scale_pixels[0],scale_pixels[1],scale_pixels[2])
    else:
        print("Error: Scale pixels needs to be a list or tuple of length 3")

def rmv_background_vectors(vectors):

    if len(vectors.shape) == 3:
        velocity = vectors[:,1,1:]
        mask = np.sum(velocity, axis=1) !=0
        masked_velocity = np.zeros((np.sum(mask),)+vectors.shape[1:])
        for j in range(vectors.shape[1]):
            for i in range(vectors.shape[2]):
                masked_velocity[:,j,i] = vectors[:,j,i][mask]

    elif len(vectors.shape) == 2:
        velocity = vectors
        mask = np.sum(velocity, axis=1) !=0
        masked_velocity = np.zeros((np.sum(mask),)+vectors.shape[1:])
        for i in range(vectors.shape[1]):
            masked_velocity[:,i] = vectors[:,i][mask]
    
    return masked_velocity,mask

# find the end point of each vector from a total_vectors array
def vector_end(vectors):

    end_point = np.zeros((vectors.shape[0],vectors.shape[2]))
    end_point[:,1:] = vectors[:,0,1:] + vectors[:,1,1:]
    end_point[:,0] = vectors[:,0,0]

    return end_point

#%% Bijoy Daga implementation of Kabsch registration, P is vector start and Q is vector end
def kabsch(P, Q):
    assert P.shape == Q.shape # matrix dimensions must match

    # Calculate the centroids of P and Q
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the points around the centroids
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Calculate the covariance matrix
    H = np.dot(Q_centered.T, P_centered)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Calculate the optimal rotation matrix
    V = Vt.T  # Transpose of Vt gives us V
    R = np.dot(V, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)

    # Calculate the translation vector
    t = centroid_Q - np.dot(centroid_P, R)

    return R, t

#%% adapted from Bijoy Daga - decomposing rigid rotation and flow
def rotation_decomp(total_vectors):
    end_vector = vector_end(total_vectors)

    # Find the optimal rotation matrix and translation vector
    R, t = kabsch(total_vectors[:,0,1:],end_vector[:,1:])

    print("Optimal rotation matrix:\n", R)
    print("Translation vector:\n", t)

    # Rotate and translate P using the rotation matrix and translation vector
    P_transformed = np.dot(total_vectors[:,0,1:],R) + t

    # Calculate R*P + t - P and flow-velocity
    rigid = P_transformed - total_vectors[:,0,1:]
    print("R*P + t - P:\n", rigid)
    flow=total_vectors[:,1,1:]-rigid

    return rigid, flow, R,t

def write_to_vtk(points,velocities,other_data,names,filename):

    raw_field = vtk_structured_grid(filename)
    piv_dims = raw_field.GetDimensions()

    filename = Path(filename)
    vtk_data = vtkStructuredGrid()

    vtk_points = vtkPoints()
    for id in range(len(points)):
        vtk_points.InsertPoint(id, points[id])
    
    vtk_data.SetPoints(vtk_points)

    
    vtk_array = VN.numpy_to_vtk(velocities)
    vtk_array.SetName(names[0])
    vtk_data.GetPointData().SetVectors(vtk_array)
    vtk_data.SetDimensions(piv_dims)

    for dat, name in zip(other_data,names[1:]):
        vtk_array = VN.numpy_to_vtk(dat)
        vtk_array.SetName(name)
        vtk_data.GetPointData().AddArray(vtk_array)

    writer = vtkStructuredGridWriter()
    writer.SetFileName(filename.parent.joinpath(filename.stem).with_suffix(f".{names[0]}.vtk"))
    writer.SetInputData(vtk_data)
    writer.Update()
    writer.Write()
    print("done")

#%% iterate over all vtk files in the PIV output to create a timeseries of vectors
def process_timeseries(input_path,interSize,overlap,pixel_dims):

    filenames = [f for f in input_path.iterdir() if f.is_file() and f.suffix == '.vtk']
    #filenames = filenames[0:79]
    print(filenames)
    with open(input_path.joinpath("registration.csv"),'w') as matrix_file:
        matrix_file.write("rotation,translation")
        for filename in filenames:
            chunks,velocities = get_vtk_vectors(filename)
            vector_start = vector_origin(chunks,interSize,overlap)
            total_vectors = combine_vectors(vector_start,velocities,filename,pixel_dims)
            total_vectors,mask = rmv_background_vectors(total_vectors)
            rigid,flow,R,t = rotation_decomp(total_vectors)
            matrix_file.write("\n")
            np.savetxt(matrix_file,R,newline=" ")
            matrix_file.write(",")
            np.savetxt(matrix_file,t,newline =" ")
            write_to_vtk(total_vectors[:,0,1:],rigid,[chunks[mask]],['rigid','chunks'],filename)
            write_to_vtk(total_vectors[:,0,1:],flow,[chunks[mask]],['flow','chunks'],filename)
            print(filename)

    return 

def view_vectors(input_path,interSize,overlap,scale_pixels=None,vector_type = 'flow'):

    if vector_type == 'unregistered':
        filenames = [f for f in input_path.iterdir() if f.is_file() and "".join(f.suffixes) == ".vtk"]
    else:
        filenames = [f for f in input_path.iterdir() if f.is_file() and "".join(f.suffixes) == "."+vector_type+".vtk" ]
    
    filenames = np.sort(filenames)
    total_vectors = []
    for filename in filenames:
        print(f'Combining vectors for visualisation for {filename}')
        #note when loading processed vtk files, the coordinates are in image dimensions 
        chunks,velocities = get_vtk_vectors(filename)
        if vector_type == 'unregistered':
            vectors = combine_vectors(chunks,velocities,filename,scale_pixels=scale_pixels)
            vectors,mask = rmv_background_vectors(vectors)
        else:
            vectors = combine_vectors(chunks,velocities,filename,scale_pixels=None)
        
        total_vectors.append(vectors)
    
    total_vectors = np.concatenate(total_vectors,axis=0)
    
    total_vectors = total_vectors[:,:,[0,3,2,1]]
    
    # viewer = napari.Viewer()
    # viewer.add_vectors(total_vectors,edge_color='white',opacity=1.0,length=3.0,edge_width=3.0,vector_style='line')

    # napari.run()

    return total_vectors

#this generates a 3d grid x3 for each timepoint where the values are the velocity at that chunk in x,y,z
def velocity_grid(filenames):
    total_vector_grid = []
    for filename in filenames:
        
        data = vtk_structured_grid(filename)
        #piv dims is the number of PIV chunks in XYZ axes. This is needed to build the vector grid. there is one vector per chunk
        piv_dims = data.GetDimensions()
        #the chunk id for each of filtered vectors is needed to know where to place each vector from the list into the grid ESPECIALLY IMPORTANT once vectors have been filtered
        chunks = VN.vtk_to_numpy(data.GetPointData().GetArray('chunks')) - 1 #adjust for python zero indexing
        chunks = chunks.astype(np.int32) #dtype changed to int to allow indexing
        #vector velocity list, X Y Z displacement for each vector
        velocities = VN.vtk_to_numpy(data.GetPointData().GetVectors())
        #build vector grid
        vector_grid = np.zeros(piv_dims)
        #stack grid so that each X, Y, Z displacement can be logged separately
        vector_grid = np.stack([vector_grid,vector_grid,vector_grid],axis=-1)

        #chunks,velocities = get_vtk_vectors(filename)
        # vector_index = np.unravel_index(range(len(velocities)),shape=(piv_dims[0],piv_dims[1],piv_dims[2]))
        # vector_index = np.stack([vector_index[0],vector_index[1],vector_index[1]],axis=-1)

        #iterate through each velocity and the values into the grid
        
        for ch in range(len(velocities)):
            vector_grid[*chunks[ch],:] = velocities[ch]
            
        total_vector_grid.append(vector_grid)

    #finally, stack each timepoint together
    return np.stack(total_vector_grid,axis=0),piv_dims
            
def generate_pseudotracks(input_path,interSize,tstart=0,tend=None,numpoints = 1000, random_points = True,voxel_dims=(0.406,0.406,2)):
    
    #get vector filenames
    filenames = [f for f in input_path.iterdir() if f.is_file() and "".join(f.suffixes) == ".flow.vtk" ]

    if tend is None:
        tend = len(filenames)+1
    print("Starting Vector Grid")
    #create a timeseries of vector grid/cubes containing the velocities, piv dims == to the shape of the vector grid for each timepoint
    vector_grid,piv_dims= velocity_grid(filenames)
    print("Created Vector Grid")
    piv_dims = np.array(piv_dims)

    #create arrays of index values for each grid dimension
    x_indices = np.linspace(0,piv_dims[0]-1,piv_dims[0])
    y_indices = np.linspace(0,piv_dims[1]-1,piv_dims[1])
    z_indices = np.linspace(0,piv_dims[2]-1,piv_dims[2])

    #open only the origins of the vectors at the first timepoint this can be used to seed the start of the Pseudotracks
    data = vtk_structured_grid(filenames[0])
    chunks_t0 = VN.vtk_to_numpy(data.GetPoints().GetData()) -1
    chunk_idx = VN.vtk_to_numpy(data.GetPointData().GetArray('chunks')) -1
    chunk_idx = chunk_idx.astype(np.int32)
    
    #create seed points randomly from witin the valid vector origins
    if random_points == True:
        rand_index = np.random.randint(0,len(chunks_t0),numpoints)
        rand_start = chunks_t0[rand_index]
        start_idx = chunk_idx[rand_index]
        x0 = rand_start[:,0]
        y0 = rand_start[:,1]
        z0 = rand_start[:,2]
        # np.random.randint(0,piv_dims[0],numpoints)
        # y0 = np.random.randint(0,piv_dims[1],numpoints)
        # z0 = np.random.randint(0,piv_dims[2],numpoints)
    else:
        #generate tracks at every vector origin at time zero. This is a lot of vectors and takes a long time to run.
        x0 = chunks_t0[:,0]
        y0 = chunks_t0[:,1]
        z0 = chunks_t0[:,2]
        numpoints = len(chunks_t0)
        start_idx = chunk_idx
        

    centroid = np.mean(chunks_t0)

    trange = tend - tstart

    trajectories = np.zeros((numpoints,trange,5),dtype=np.float64)
    velocity = []
    direction = []
    for p in range(numpoints):
        # in order to get the vectors in the correct direction it is the vector origin that is associated with the displacement!!!!! this is the same as the vector fields
        print(f"Starting track {p} out of {numpoints}. {np.round(((p-1)/numpoints)*100,decimals=2)} % of the pseudotracks complete so far. ")
        #for each point the trajectories begin at the seed location
        trajectories[p,0,0:3] = [x0[p],y0[p],z0[p]]
        # we index the grid of velocities to get the displacement at that point
        displacement = vector_grid[0,start_idx[p,0],start_idx[p,1],start_idx[p,2]]
        #calculate other props
        trajectories[p,0,3] = np.sqrt(np.sum(displacement**2))
        trajectories[p,0,4] = norm_dotprod(trajectories[p,0,0:3]-centroid,displacement)
        #we then go through each subsequent time point
        for t in range(tstart+1,tend):
            #adding a new point to the trajectories each time, by adding the displacement to the previous timepont
            trajectories[p,t,0:3] = trajectories[p,t-1,0:3] + displacement #displacement/interSize

            #this cuts the trajectory if if it extends beyond the bounds of the grid
            # if (trajectories[p,t,0:3]/piv_dims > 1).any() or (trajectories[p,t,0:3]/piv_dims < 0).any():
            #     break
            
            #the next displacement is then calculated by interpolating nearby trajectories
            #grid_scale = (interSize,interSize,int(np.round(interSize/(2/0.406))))
            if t != (tend-1):
                sample_point = ((trajectories[p,t,0:3] / voxel_dims)/interSize)
                displacement = interpn((x_indices,y_indices,z_indices),vector_grid[t],sample_point,bounds_error=False,fill_value=0,method='linear')[0]
                
                trajectories[p,t,3] = np.sqrt(np.sum(displacement**2))
                trajectories[p,t,4] = norm_dotprod(trajectories[p,t,0:3]-centroid,displacement)

    trajectories_table = np.zeros((numpoints*trange,7))
    trajectories_table[:,0] = np.concatenate([[i] * trange for i in range(numpoints)])
    trajectories_table[:,1] = np.tile(range(tstart,tend),reps=numpoints)
    trajectories_table[:,2:] = np.concatenate(trajectories)
    
    output_path = input_path.joinpath("pseudotracks.csv") # "\\".join(input_path.split('\\')[:-1]+["pseudotracks.csv"])

    pseudotracks = pd.DataFrame(trajectories_table,columns=['track_id','time','x','y','z','displacement','radial_alignment'])



    pseudotracks.to_csv(output_path,index=None)

    print("Saved Pseudotracks")
    
    return pseudotracks #,grid_coords

def norm_dotprod(vector1, vector2):
    norm_vector1 = vector1/np.sqrt(np.sum(vector1**2))
    norm_vector2 = vector2/np.sqrt(np.sum(vector2**2))

    return np.dot(norm_vector1,norm_vector2)

# print(len(velocity),len(direction),len(pseudotracks))
# pseudotracks['velocity'] = velocity
# pseudotracks['radial_alignment'] = direction


# centroid = np.mean(pseudotracks[pseudotracks.time == 0].loc[:,['x','y','z']])
# pseudotracks.set_index(['track_id','time'],inplace=True,drop=False)
# pseudotracks['orientation'] = np.nan
# for trk_id in np.unique(pseudotracks.track_id):
#     track = pseudotracks.loc[trk_id]
#     for t in range(int(track.time.min()+1),int(track.time.max())):

        #pseudotracks.loc[trk_id,t] = np.sqrt(np.sum((track.loc[t,['x','y','z']] - track.loc[t,['x','y','z']])**2))
# ps = generate_pseudotracks(Path("C:\\Users\\User\\OneDrive - University of Cambridge\\scripts\\vector_field_analysis\\expzacy0032\\piv_output"),40,tstart=0,tend=None,numpoints = 300, random_points = True)
# print(ps)
# data = vtk_structured_grid(Path("D:\\Users\\Alice\\pescoid\\20250214_144831_expzacy0039\\Position 1_Settings 1_fused\\expzacy0039_pos1_piv_output\\t00001_s00.flow.vtk"))
# chunks = VN.vtk_to_numpy(data.GetPointData().GetArray('chunks')) -1
# rand_index = np.random.randint(0,len(chunks),10)
# rand_start = chunks[rand_index]
# print(rand_start)
# print(rand_start[:,0])

# generate_pseudotracks(Path("D:\\Users\\Alice\\pescoid\\20250214_144831_expzacy0039\\Position 1_Settings 1_fused\\expzacy0039_pos1_piv_output"),(40,40,8),tstart=0,tend=None,numpoints = 1000, seed_points = None)


# def vector_chunks(vector_start,interSize,overlap):

#     x_chunk = (vector_start[:,0] + (interSize[0]/2))/ interSize[0] #+ ( + (overlap / vector_start[:,0])
#     y_chunk =  (vector_start[:,1] + (interSize[1]/2))/ interSize[1]# + (overlap / vector_start[:,1])
#     z_chunk =  (vector_start[:,2] + (interSize[2]/2))/ interSize[2]# + (overlap / vector_start[:,2])

#     # x_chunk = (vector_start[:,0] / interSize[0]) + (interSize[0]/2) + (overlap / vector_start[:,0])
#     # y_chunk = (vector_start[:,1] / interSize[1]) + (interSize[1]/2) + (overlap / vector_start[:,1])
#     # z_chunk = (vector_start[:,2] / interSize[2]) + (interSize[2]/2) + (overlap / vector_start[:,2])

#     return np.stack((x_chunk,y_chunk,z_chunk),axis=1)


# marta piv
# process_timeseries(Path("G:\\Viventis_LS2\\20241203_155618_GV55_d03_EzrCA_v3_fused\\Position 2_Settings 1_fused\\piv_output"),(60,60,12),0,(0.406,0.406,2))
# view_vectors(Path("G:\\Viventis_LS2\\20241203_155618_GV55_d03_EzrCA_v3_fused\\Position 2_Settings 1_fused\\piv_output"),(60,60,12),0,(0.406,0.406,2))

# grid,dims = velocity_grid([Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.flow.vtk")])
# print(grid[0,20,25,1])

# data = vtk_structured_grid(Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.flow.vtk"))
# print( VN.vtk_to_numpy(data.GetPointData().GetArray('chunks'))-1)
# chunks,velocities = get_vtk_vectors("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.flow.vtk")
# print(velocities)
# print(vector_chunks(vector_origin(chunks,(40,40,8),10),(40,40,8),10))
# print(type(data.GetDimensions()))

# print(VN.vtk_to_numpy(grid.GetPoints().GetData()))

# process_timeseries(Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output"),(40,40,8),0,(0.406,0.406,2))
#view_vectors(Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output"),(40,40,8),0,(0.406,0.406,2))
# generate_pseudotracks(Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output"),tstart=0,tend=67,numpoints=1000,interSize=(40,40,8))
# pseudo_tracks = pd.read_csv("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output"+"\\pseudotracks.csv",header=None)
# pseudo_tracks.columns = ["track_id","time","x","y","z"]
# viewer = napari.Viewer()





# viewer.add_tracks(pseudo_tracks.loc[:,['track_id','time','z','y','x']],name='pseudotracks')

# napari.run()

# def write_to_vtk(arrays,names,filename):

#     data = vtk_structured_grid(filename)
    
#     for array,name in zip(arrays,names):
#         vtk_array = VN.numpy_to_vtk(array)
#         vtk_array.SetName(name)
#         data.GetPointData().AddArray(vtk_array)

#     writer = vtkStructuredGridWriter()
#     writer.SetFileName("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\test.vtk")
#     writer.SetInputData(data)
#     writer.Update()
#     writer.Write()
#     print("done")


# the next two lines commented by guillermo on 25/11/24
#data = vtk_structured_grid(Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.vtk"))
#print(data.GetDimensions(dims[3]))


# velocities,dims,grid,chunks = get_vtk_vectors(Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.vtk"))
# origins = vector_origin(chunks,(40,40,8),0)
# total_vectors = combine_vectors(origins,velocities,Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.vtk"),scale_pixels=(0.406,0.406,2))
# total_vectors = rmv_background_vectors(total_vectors)
# rigid,flow = rotation_decomp(total_vectors)
# write_to_vtk(total_vectors[:,0,1:],[rigid,flow],['rigidbody_velocities','flow_velocities'],Path("D:\\Users\\Alice\\viventis_processed\\20241030_195501_expzacy0032\\piv_output\\t00001_s00.vtk"))

# 
# ends = vector_end(total_vectors)
# R,t = kabsch(total_vectors[:,0,1:],ends[:,1:])
# print(R,t)
# print(total_vectors.shape)
# print(velocities.shape)

#print(VN.vtk_to_numpy(data.GetPointData().GetArray("directions")))


# tracks = generate_pseudotracks("D:\\Users\\Imen\\RA\\exp12\\piv_output\\*s00*",tstart=0,tend=142,numpoints=100,interSize=(52,52,7)) #generate_pseudotracks("D:\\Users\\Alice\\viventis_raw\\20240702_181242_expzacy0001_fused\\Position 1_Settings 1_fused\\piv_output_expzacy0001\\*561*",tstart=0,tend=78,numpoints=1000,interSize=(40,40,8))
# # vectors = complete_vectors("D:\\Users\\Alice\\Viventis_raw\\20240702_181242_expzacy0001_fused\\Position 1_Settings 1_fused\\piv_output\\*561*",(40,40,8),0)

# viewer = napari.Viewer()

# #viewer.open("D:\\Users\\Alice\\20231214_162945_Pescoid_Tbx6GFP_H2BmCherrymRNA_fused.zarr",plugin='napari-ome-zarr',contrast_limits = (0,2**16),scale=(1,5,1,1),colormap='cyan')

# viewer.add_tracks(tracks.loc[:,['track_id','time','z','y','x']])
# # viewer.dims.set_point(0,0)
# # viewer.dims.set_point(1,123)
# # viewer.add_vectors(vectors,edge_color='white',scale=(1,5,1,1),opacity=1.0,length=3.0,edge_width=3.0,vector_style='line')


# napari.run()


# def vector_origin(piv_dims,interSize,overlap):
	
# 	#chunk_num = img_dims/interSize

#     w,h,d = piv_dims
    
#     #origin_matrix = np.zeros(vector_end.shape)
#     origin_list = np.zeros(((h*w*d),3))
#     count = 0
#     for z in range(d):
#         for y in range(h): 
#             for x in range(w):
#                 coord = np.array((x,y,z))
                
#                 origin = [((coord[0] +1) * interSize[0]) - (interSize[0]/2) - (overlap * coord[0]),
#                           ((coord[1] +1) * interSize[1]) - (interSize[1]/2) - (overlap * coord[1]),
#                           ((coord[2] +1) * interSize[2]) - (interSize[2]/2) - (overlap * coord[2])]
                
#                 origin_list[count] = origin
#                 count += 1
#                 #origin_list.append(origin)
                
#     return origin_list#.astype(np.uint64)




# def lazy_vectors(input_path,interSize,overlap):
#     filenames = sorted(glob(input_path), key=alphanumeric_key)
#     filenames = filenames[0:230]
    
#     sample = vector_timeseries(filenames[0],interSize,overlap)
    
#     lazy_imread = delayed(vector_timeseries)  # lazy reader
#     lazy_arrays = [lazy_imread(fn,interSize,overlap) for fn in filenames]
#     dask_arrays = [
#         da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
#         for delayed_reader in lazy_arrays
#         ]
    
#     stack = da.concatenate(dask_arrays, axis=0)

#     return stack



# def full_vector_grid(input_path,tstart,tend):
#     filenames = sorted(glob(input_path), key=alphanumeric_key)
#     if tend == -1:
#         filenames = filenames[tstart:]
#         tend=len(filenames)
#     else:
#         filenames = filenames[tstart:tend]

#     total_vector_grid = []
#     for filename in filenames:
#         vector_end,piv_dims,vector_grid,chunks = get_vtk_vectors(filename)
#         vector_grid = np.stack([vector_grid,vector_grid,vector_grid],axis=-1)

#         vector_index = np.unravel_index(range(len(vector_end)),shape=(piv_dims[2],piv_dims[1],piv_dims[0]))
#         vector_index = np.stack([vector_index[2],vector_index[1],vector_index[0]],axis=-1)
        
#         for ch in range(len(vector_end)):
#             vector_grid[*vector_index[ch],:] = vector_end[ch]
            
#         total_vector_grid.append(vector_grid)
    
    
#     return np.stack(total_vector_grid,axis=0),piv_dims,chunks,tend 



