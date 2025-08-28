'version: 1.0.0'
from multiprocessing import freeze_support

from magicgui import magicgui
from pathlib import Path
import tiff_conversion
import logging
import napari
from magicgui.widgets import Container, create_widget
import numpy as np
import xml.etree.ElementTree as ET

#this is the main file for processing lightsheet data from the viventis
# the code here runs napari and puts in several magicgui widgets which call each of the scripts for different processing options

# this logger is created for logging the progress of tiff to hdf5 conversion
logger = logging.getLogger(__name__)

#this is the widget for tiff to hdf5 conversion
@magicgui(ExperimentPath={"mode": "d", "label": "Choose a directory"},call_button='Begin Conversion')
def tiff2BDVConversion(ExperimentPath=Path("Choose Experiment folder"),Input_Key:str="*Fused*",Time_Start:int=0,Time_End:int=-1,CRSID:str='ddzs2')->str:
    ''' This function and the associated magicgui widget should be used to convert viventis LS2 ome.tiff files to an HDF5/XML pair
        For each position within the given viventis Experiment Folder the following will happen...
        First it will create a copy of the data in HDF5/XML format without a multiresolution pyramid and compressed using ___ level (as of June 2025). 
        This will be stored in a mirrored location on the 225 server. If you no longer need this data delete it. Eventually it will be sent to the cold store for long term storage.
        It will then uncompress that copy and duplicate the data into a normal GZIP compressed HDF5/XML pair that can be read by FIJI's bigdataviewer and Imaris
        Finally, it will create a multiresolution pyramid of the data in the same file for rapid and easy viewing
        The XML file will be populated with all the metadata from the ome.companion files and settings folders produced by the viventis LS2
        The final HDF5/XML file will be saved in the same experiment folder as the original data.
        
        Note: maximum projections and single slice brightfield images are ignored and not part of this conversion
        
        ExperimentPath: pathlib.Path object (handles different filesystem conventions for paths) pointing to the Viventis LS2 experiment data for conversion
        Input_Key: string containing Unix style pattern for which to recognise individual files. Use asterisk for wildcard. This parameter allows you to convert individual channels. If all channels needed then leave as default (defulat "*Fused*")
        Time_Start: integer timepoint at which to start conversion (default 0)
        Time_End: integer timepoint at which to end conversion (default -1 : all timepoints after start)
        CRSID: string containing your CRSID eg the first part of your email, this is added into the metadata so we know who made the conversion '''
    
    
    # if time end is set to -1 then run through all timepoints
    if Time_End == -1:
        Time_End = None 

    tiff_conversion.experiment2bdv(ExperimentPath,Input_Key,ExperimentPath,Time_Start,Time_End,CRSID,'gzip','int16',stor_location=Path("Y:\\Room225_SharedFolder\\ViventisLS2_data"),pyramid_levels = ((1, 4, 4),(2, 8, 8),(4, 16, 16)),chunk_dims=((4, 250, 250),(4, 250, 250), (4, 250, 250), (4, 250, 250)))

    return



class CroppingData(Container):
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__()
        self._viewer = viewer
        #widgets
        self._filedialog = create_widget(widget_type = 'FileEdit',label ='Select Position to Crop',options=dict(mode='d'))
        self._inputkey = create_widget(widget_type='LineEdit',label='Input Key',value='*Fused*')
        self._channel = create_widget(widget_type='LiteralEvalLineEdit',label='Channel to Project',value='0')
        self._timestep = create_widget(widget_type='LiteralEvalLineEdit',label='TimeStep',value='10')
        
        self._projectButton = create_widget(widget_type = "PushButton",options=dict(text='Max Project Data'))
        self._yxdimsButton = create_widget(widget_type = "PushButton",options=dict(text='Capture YX Dimensions'))
        self._zydimsButton = create_widget(widget_type = "PushButton",options=dict(text='Capture ZY Dimensions'))
        self._finishcropButton = create_widget(widget_type = "PushButton",options=dict(text='Save Crop Dimensions'))

        #callbacks
        self._projectButton.changed.connect(self.project_and_display)
        self._yxdimsButton.changed.connect(self.get_dims)
        self._zydimsButton.changed.connect(self.get_dims)
        self._finishcropButton.changed.connect(self.save_crop)

        #crop values
        self.scale_fac = None
        self.x_crop = None
        self.y_crop = None
        self.z_crop = None

        self.extend([self._filedialog,self._inputkey,self._channel,self._timestep,self._projectButton,self._yxdimsButton,self._zydimsButton,self._finishcropButton])

    def project_and_display(self):
        print('Starting Max Projection of data')
        
        yx_max_final, zy_max_final,pixel_meta = tiff_conversion.project_time_series(self._filedialog.value,self._inputkey.value,self._timestep.value,self._channel.value)
        #yx_max_final = tiff.imread(Path("D:\\Users\\Dillan\\for_mounting_measure\\20231124_105814_Experiment\\yx_max_final.tif"))
        print('Finished Projecting Data')
        
        self.scale_fac = float(pixel_meta['PhysicalSizeZ'])/float(pixel_meta['PhysicalSizeY'])
        
        default_square = np.array([[ 0,  0],
                                [ 200, 0],
                                [200, 200],
                                [0, 200]])
        self._viewer.add_image(zy_max_final,name='ZY_Projection',scale=(self.scale_fac,1))
        self._viewer.add_shapes(default_square,shape_type='rectangle',face_color=[[1,1,1,0]],edge_color=[[1,0,0,1]],edge_width=[5.0],text=['Use this rectangle to get crop bounds'],name='ZY_CropBox')
        self._viewer.layers['ZY_CropBox'].mode ='SELECT'
        
        self._viewer.add_image(yx_max_final,name='YX_Projection')
        self._viewer.add_shapes(default_square,shape_type='rectangle',face_color=[[1,1,1,0]],edge_color=[[1,0,0,1]],edge_width=[5.0],text=['Use this rectangle to get crop bounds'],name='YX_CropBox')
        self._viewer.layers['YX_CropBox'].mode ='SELECT'

        return
    
    def get_dims(self):
        
        layer = self._viewer.layers.selection.active
        if type(layer) == napari.layers.shapes.shapes.Shapes:
            box = layer.data[0].astype(int)
            crop_start = box[0]
            crop_end = box[2]
            print(f'These are the crop dimensions for {layer.name}')
            if layer.name == 'YX_CropBox':
                print(f'Y-Dimension start:end {crop_start[0]-10}:{crop_end[0]+10}')
                self.y_crop = [crop_start[0]-10,crop_end[0]+10]
                print(f'X-Dimension start:end {crop_start[1]-10}:{crop_end[1]+10}')
                self.x_crop = [crop_start[1]-10,crop_end[1]+10]
            if layer.name == 'ZY_CropBox':
                print(f'Z-Dimension start:end {crop_start[0]-10}:{crop_end[0]+10}')
                self.z_crop = [crop_start[0]-10,crop_end[0]+10]
        else:
            print('Select a layer with the name Cropbox to calculate the cropping dimensions')

    def save_crop(self):
        assert(self.x_crop != None), "No crop in x-dimension. Use the YX_cropbox to capture YX-dimensions before saving"
        assert(self.y_crop != None), "No crop in y-dimension. Use the YX_cropbox to capture YX-dimensions before saving"
        assert(self.z_crop != None), "No crop in z-dimension. Use the ZY_cropbox to capture ZY-dimensions before saving"
        
        self.x_crop = tiff_conversion.boundary_zero(dim_list = self.x_crop)
        self.y_crop = tiff_conversion.boundary_zero(dim_list = self.y_crop)
        self.z_crop = tiff_conversion.boundary_zero(dim_list = self.z_crop)

        fp = self._filedialog.value.joinpath("CropDimensions.xml")
        assert(fp.exists() == False),"You already have cropped dimensions saved in this folder, delete them a try again"

        crop_element = ET.Element('CroppingDimensions')
        crop_element.text = '\n    '

        x_elem = ET.SubElement(crop_element,'XDims')
        x_elem.attrib = {'x_start':f'{self.x_crop[0]}','x_end':f'{self.x_crop[1]}'}
        x_elem.tail = '\n    '
        y_elem = ET.SubElement(crop_element,'YDims')
        y_elem.attrib = {'y_start':f'{self.y_crop[0]}','y_end':f'{self.y_crop[1]}'}
        y_elem.tail = '\n    '
        z_elem = ET.SubElement(crop_element,'ZDims')
        z_elem.attrib = {'z_start':f'{int(np.round(self.z_crop[0]/self.scale_fac))}','z_end':f'{int(np.round(self.z_crop[1]/self.scale_fac))}'}
        z_elem.tail = '\n    '

        metadata = ET.ElementTree(crop_element)
        metadata.write(fp,xml_declaration=True, encoding='utf-8', method="xml")
        
        assert(fp.exists()), "The cropped dimensions where not written to file, something interrupted."
        print(f"Cropping Dimensions have been saved at {fp}")
        self.x_crop = None
        self.y_crop = None
        self.z_crop = None


@magicgui(FilePath={"label": "Choose a file"},call_button='Begin Conversion')
def retrievingSTORedData(FilePath=Path("Select File")) -> str:
    
    active_data_drive = Path("Z:\\Room224_SharedFolder") 
    FilePath = FilePath.parent.joinpath(Path(FilePath.stem).stem)
    active_data_path = active_data_drive.joinpath(*FilePath.parts[3:])
    active_data_path.parent.mkdir(parents=True, exist_ok=True)

    tiff_conversion.storage2bdv(active_data_path,stor_location=Path("Y:\\Room225_SharedFolder\\ViventisLS2_data\\"))

    return


if __name__ == '__main__':
    freeze_support()
    
    print('THANK YOU FOR CHOOSING TO PROCESS YOUR DATA WITH US...\n Please wait while the napari window opens, this can sometimes take a while')

    viewer = napari.Viewer()

    widge = CroppingData(viewer)
    viewer.window.add_dock_widget(widge,name="Cropping Lightsheet Data")

    viewer.window.add_dock_widget(tiff2BDVConversion,name="Convert Lightsheet Experiment")

    viewer.window.add_dock_widget(retrievingSTORedData,name="Retrieve STORed File")


    napari.run()


