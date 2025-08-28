'version: 1.0.0'
from magicgui import magicgui
from pathlib import Path
import logging
import napari
import subprocess
import pandas as pd
import quickpiv
import ast
import numpy as np
from magicgui.widgets import Container, create_widget
from napari.utils.color import ColorValue
from napari._pydantic_compat import Field
from matplotlib import colors
from matplotlib import cm
#this is the main file for processing lightsheet data from the viventis
# the code here runs napari and puts in several magicgui widgets which call each of the scripts for different processing options
print('THANK YOU FOR CHOOSING TO PROCESS YOUR DATA WITH US...\n Please wait while the napari window opens, this can sometimes take a while')

# this logger is created for logging the progress of tiff to hdf5 conversion
logger = logging.getLogger(__name__)

@magicgui(Input_Path={"mode": "d", "label": "Choose a directory"},call_button="Run 3D PIV")
def run_piv(Input_Path = Path("Choose Folder to Analyse with PIV"),key: str="Settings1.h5",is_hdf5: bool = True, hdf5_target_channel: str=0,PIV_WindowSize: str=40,PIV_SearchMargin: str=20,Voxel_Dimensions: str="0.406,0.406,2")-> str:
    ''' This function and associated magicgui widget runs 3D particle Image Velocimetry on image data
        The function opens Julia through the command line and runs the Julia scripts inside batch_quickpiv. These functions are for calculating PIV across our timelapse data. 
        The code actually calculating the 3D PIV is from: https://doi.org/10.1186/s12859-021-04474-0 and the original source code can be found at: https://github.com/Marc-3d/quickPIV 
        The PIV code uses Julia because its fast and can be run easily in parallel
        Note: Some minor changes were made to the original source code so try to keep using the Honey Badger version where possible

        Input_Path: pathlib.Path object pointing to the Experiment folder containing the data on which to run PIV
        key: string used as pattern to select which files are to be analysed. Currently 3DPIV has to be run one at a time for each position within an experiment, specify the HDF5 name here to select different positions for analysis. Not Unix style so DONT use asterisk. This can be used to select a specific channel is dealing with tiff files. (default: "Settings1.h5")
        is_hdf5: boolean, for determining how to the load the files of interest (default: True)
        hdf5_target_channel: integer (converted to string in the background), the number of the channel that you wish to analyse with PIV, 0 based. Channel order will be in alphabetical/numerical order, do not put the channel name here (Default: "0")
        PIV_WindowSize: integer (converted to string in the background), the size of the window used for the PIV analysis in pixels. This can be roughly the size of the particles of interest. The value will be used to create a cube (X,Y,Z) with account for Z-anisotropy. (default: 40) 
        PIV_SearchMargin: integer (converted to string in the background), the size by which to increase the PIV_WindowSize when searching for cross-correllation in pixels. this can be roughly the average displacement of the particles.The value will be used to create a cube (X,Y,Z) with account for Z-anisotropy (Default: 20)
        Voxel_Dimensions: string of three consecutive floats separated by commas. The size of a single voxel in the image data. this will vary depending on the Z-step of the imaging run and the magnification of the objective used. (Defaul: "0.406,0.406,2")
    '''
    # note all arguments to subprocess must be strings so the boolean needs converting. Python bool is True/False, Julia bool is true/false
    Input_Path = str(Input_Path) + "\\"
    if is_hdf5 == True:
        is_hdf5 = "true"
    else:
        is_hdf5 = 'false'

    subprocess.run(["julia","--threads=100","--project=C:\\Users\\Public\\data_processing_scripts\\batch_quickpiv",
                    "C:\\Users\\Public\\data_processing_scripts\\batch_quickpiv\\commandline_julia.jl",Input_Path,key,is_hdf5,hdf5_target_channel,
                    PIV_WindowSize,PIV_SearchMargin,Voxel_Dimensions],shell=True)
    return

class visualise_vectors(Container):
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._filedialog = create_widget(widget_type = 'FileEdit',label ='Choose Folder Containing PIV Results',options=dict(mode='d'))
        self._pivwindowsize = create_widget(widget_type='SpinBox',label='PIV Window Size',value='40')
        self._voxeldims = create_widget(widget_type='LiteralEvalLineEdit',label='Voxel Dimensions',value='0.406,0.406,2')

        self._registerButton = create_widget(widget_type = "PushButton",options=dict(text='Register PIV Vector Fields'))
        self._vectorDropdown = create_widget(widget_type="ComboBox",label='Select vector type to visualise',options=dict(choices = ['flow','rigid','unregistered']))
        self._propsDropdown = create_widget(widget_type="ComboBox",label='Color Vector by Property',options=dict(choices = ['none','magnitude'],value = 'none'))
        self._visualiseButton = create_widget(widget_type="PushButton",options=dict(text='Visualise Vector Fields'))

        self._registerButton.changed.connect(self.register_piv)
        self._visualiseButton.changed.connect(self.visualise_vectors)

        self.extend([self._filedialog,self._pivwindowsize,self._voxeldims,self._registerButton,self._vectorDropdown,self._propsDropdown,self._visualiseButton])

    def register_piv(self):
        
        PIV_window_size = (self._pivwindowsize.value,self._pivwindowsize.value,int(np.round(self._pivwindowsize.value/(self._voxeldims.value[2]/self._voxeldims.value[0]))))
        pivoutput_files = [f.name for f in self._filedialog.value.iterdir()]
        overlap = 0
        
        assert (any('flow' in f for f in pivoutput_files) == False), "The vector fields have already been registered. Remove all 'flow' or 'rigid' vtk files before registering again"
       
        print('Beginning Kabsch registration of PIV vector fields')
        quickpiv.process_timeseries(self._filedialog.value,PIV_window_size,overlap,self._voxeldims.value)

    def visualise_vectors(self):
        PIV_window_size = (self._pivwindowsize.value,self._pivwindowsize.value,int(np.round(self._pivwindowsize.value/(self._voxeldims.value[2]/self._voxeldims.value[0]))))
        overlap = 0
        vector_field = quickpiv.view_vectors(self._filedialog.value,PIV_window_size,overlap,vector_type=self._vectorDropdown.value,scale_pixels = self._voxeldims.value)

        if self._propsDropdown.value == 'magnitude':
            magnitude = np.sqrt(np.sum((vector_field[:,1,1:] - vector_field[:,0,1:])**2,axis=1))
            self._viewer.add_vectors(vector_field,edge_color='magnitude',opacity=1.0,length=1.0,edge_width=1.0,vector_style='line',features = pd.DataFrame(magnitude,columns=['magnitude']))
        else:
            self._viewer.add_vectors(vector_field,edge_color='white',opacity=1.0,length=1.0,edge_width=1.0,vector_style='line')

        new_range = napari.components.dims.RangeTuple(start=np.float64(0.0),stop=np.float64(np.unique(vector_field[:,:,0]).shape[0]-1),step=np.float64(1.0))
        self._viewer.dims.range = (new_range,)+self._viewer.dims.range[1:]


# @magicgui(PIVOutput_Folder={"mode": "d", "label": "Choose a directory"},call_button='Process PIV Output')
# def register_piv(PIVOutput_Folder=Path("Choose Folder Containing PIV Results"), PIV_window_size: int=40, voxel_dims: str="0.406,0.406,2", vector_type: str='flow') -> napari.layers.Vectors:
#     voxel_dims = ast.literal_eval(voxel_dims)
#     PIV_window_size = (PIV_window_size,PIV_window_size,int(PIV_window_size/(voxel_dims[2]/voxel_dims[0])))
#     pivoutput_files = [f.name for f in PIVOutput_Folder.iterdir()]
#     overlap = 0

#     assert vector_type == 'flow' or vector_type =='rigid', 'vector type must be one of the two decomposed parts of the vector field. Either "flow" for the local movement or "rigid" for the global movement.'

#     if any(vector_type in f for f in pivoutput_files):
#         vector_field = quickpiv.view_vectors(PIVOutput_Folder,PIV_window_size,overlap,vector_type=vector_type)
#     else:
#         print('Beginning Kabsch registration of PIV vector fields')
#         quickpiv.process_timeseries(PIVOutput_Folder,PIV_window_size,overlap,voxel_dims)
#         vector_field = quickpiv.view_vectors(PIVOutput_Folder,PIV_window_size,overlap,vector_type=vector_type)

#     magnitude = np.sqrt(np.sum((vector_field[:,1,1:] - vector_field[:,0,1:])**2,axis=1))

#     return napari.layers.Vectors(vector_field,edge_color='magnitude',opacity=1.0,length=1.0,edge_width=1.0,vector_style='line',features = pd.DataFrame(magnitude,columns=['magnitude']))

@magicgui(PIVOutput_Folder={"mode": "d", "label": "Choose a directory"},call_button='Process PIV Output')
def pseudotrack_widget(PIVOutput_Folder=Path("Choose Folder Containing PIV Results"), PIV_window_size: int=40, voxel_dims: str="0.406,0.406,2", pstrk_random: bool=True, pstrk_points: int=1000) -> napari.layers.Tracks:
    # chunk_size_xy: int=52,chunk_size_z: int=7
    voxel_dims = ast.literal_eval(voxel_dims)
    PIV_window_size = (PIV_window_size,PIV_window_size,int(PIV_window_size/(voxel_dims[2]/voxel_dims[0])))
    pivoutput_files = [f.name for f in PIVOutput_Folder.iterdir()]
    overlap = 0

    if any("pseudotracks" in f for f in pivoutput_files):
        print("Pseudotracks file already exists at this location. If this was a mistake remove any files containing the word 'pseudotracks'. Loading in pseudotracks...")
        pseudo_tracks = pd.read_csv(str(PIVOutput_Folder)+"\\pseudotracks.csv")
        pseudo_tracks = pseudo_tracks[pseudo_tracks.displacement != 0] #where a pseudotrack has nowhere to go the interpolation fills the displacement with zeros, these need to be removed to visualise properly, radial_alignment cannot have nans
    else:
        print("Converting VectorField to Pseudotracks")
        pseudo_tracks = quickpiv.generate_pseudotracks(PIVOutput_Folder,tstart=0,tend=None,numpoints=pstrk_points, random_points = pstrk_random,interSize=PIV_window_size,voxel_dims=voxel_dims) 
        #pseudo_tracks.columns = ["track_id","time","x","y","z","velocity","radial_alignment"]
        pseudo_tracks = pseudo_tracks[pseudo_tracks.displacement != 0] #where a pseudotrack has nowhere to go the interpolation fills the displacement with zeros, these need to be removed to visualise properly, radial_alignment cannot have nans

    return napari.layers.Tracks(pseudo_tracks.loc[:,['track_id','time','z','y','x']],name='pseudotracks',features = pseudo_tracks,color_by='displacement',colormap='viridis')

# class color_bar(napari.components.overlays.base.CanvasOverlay):
#     colored: bool = True
#     color: ColorValue = Field(default_factory=lambda: ColorValue([1, 0, 0, 1]))
#     box=True
    
# def generate_colorbar(data,cmap):
    
#     #take the min and max vals of your data and normalize the range
#     norm = colors.Normalize(vmin=data.min(), vmax=data.max())

#     #scale the color map to fit the range of the normalized data
#     smap = cm.ScalarMappable(norm = norm, cmap = cmap)
#     return smap


viewer = napari.Viewer()


viewer.axes.visible = True
viewer.axes.colored = False
viewer.scale_bar.visible = True
viewer.scale_bar.ticks = False
viewer.scale_bar.font_size = 20.0
viewer.scale_bar.length = 100
viewer.scale_bar.unit = "µm"

# colorbar = color_bar()
# colorbar.visible = True
# colorbar.position =  'bottom_left'
# viewer._overlays.update({'colorbar':colorbar})

viewer.window.add_dock_widget(run_piv,name="Run 3D PIV")

widge = visualise_vectors(viewer)
viewer.window.add_dock_widget(widge,name="Register and View PIV Vector Fields")

viewer.window.add_dock_widget(pseudotrack_widget,name ="Create Psuedotracks from Vector Fields")

viewer.dims.axis_labels = ['Time','Z','Y','X']

napari.run()







