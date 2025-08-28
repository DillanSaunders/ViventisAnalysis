import tifffile as tiff
import xml.etree.ElementTree as ET
import time
from dask import delayed
import dask.array as da
from pathlib import Path
import shutil
import numpy as np
import h5py
import hdf5plugin
import npy2bdv
import json
from multiprocessing import Pool
from itertools import repeat
import logging
from skimage.exposure import rescale_intensity

logger = logging.getLogger(__name__)


def read_metadata(input_path):

    metadata_path = input_path.glob("ome-tiff.companion*")
    with open(sorted(metadata_path)[0],"r") as file:
        
        metadata = file.read()

    root = ET.fromstring(metadata)

    return root[0][0].attrib,root

def read_crop_dims(input_path,pixel_meta):
    cd_xml = input_path.joinpath("CropDimensions.xml")
    if cd_xml.exists():
        cd_tree = ET.parse(cd_xml,parser = ET.XMLParser(encoding="utf-8"))
        root = cd_tree.getroot()
        x_dims = root[0].attrib
        y_dims = root[1].attrib
        z_dims = root[2].attrib

        return x_dims,y_dims,z_dims
    else:

        x_dims = dict(x_start = '0',x_end = pixel_meta['SizeX'])
        y_dims = dict(y_start = '0',y_end = pixel_meta['SizeY'])
        z_dims = dict(z_start = '0',z_end = pixel_meta['SizeZ'])

        return x_dims,y_dims,z_dims


def lazy_loading(input_path,input_key,tstart=0,tend = None):
    # code adapted from https://napari.org/stable/tutorials/processing/dask.html
    filenames = sorted(input_path.glob(input_key))
    # read the first file to get the shape and dtype
    # ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
    #sample = tiff.imread(filenames[0])

    # reads from ome meta_data file instead
    pixel_metadata,all_metadata = read_metadata(input_path)

    if tend == None:
        tend = int(pixel_metadata['SizeT'])
    
    lazy_imread = delayed(tiff.imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn,key = range(0,int(pixel_metadata['SizeZ']))) for fn in filenames]
    
    dask_arrays = []
    for i in range(len(lazy_arrays)):
        
        if i%int(pixel_metadata['SizeC']) == 0:
            
            channel_arrays = [
                da.from_delayed(lazy_arrays[j],shape=(int(pixel_metadata['SizeZ']),int(pixel_metadata['SizeY']),int(pixel_metadata['SizeX'])), dtype=pixel_metadata['Type'])
                for j in range(i,i+int(pixel_metadata['SizeC']))
            ]
            dask_arrays.append(da.stack(channel_arrays,axis=0))
            
        
    # # Stack into one large dask.array
    
    stack = da.stack(dask_arrays[tstart:tend], axis=0)
    #stack.shape  # (ntimepoints,nchannels, nx, ny, nz)
    
    
    return stack,pixel_metadata,all_metadata

def project_slice(stack,channel,yx_dir,zy_dir,index):
    print(f'Projecting timepoint {index}.')
    yx_slice = np.max(stack[index,channel,...].compute(),axis=0)
    tiff.imwrite(yx_dir.joinpath(f"yx_max_t{str(index).zfill(4)}.tif"),yx_slice)
    
    zy_slice = np.max(stack[index,channel,...].compute(),axis=2)
    tiff.imwrite(zy_dir.joinpath(f"zy_max_t{str(index).zfill(4)}.tif"),zy_slice)


def project_time_series(input_path,input_key,tp_step, channel):
    
    #create a dask stack for the image data
    stack, pixel_meta, meta = lazy_loading(input_path,input_key)
    print("Image data lazy loaded")
    
    #create the folders to save the projections. this is to ensure that time isnt wasted for large movies
    yx_dir = input_path.joinpath("0_yx_projection_DELETEME") #note: they are numbered 0 and 1 so they will be at the top of the list of directory contents
    yx_dir.mkdir(exist_ok=True)
    zy_dir = input_path.joinpath("1_zy_projection_DELETEME")
    zy_dir.mkdir(exist_ok=True)
    
    
    tp_index = np.array(range(0,stack.shape[0],tp_step))

    if (len([*zy_dir.glob("*.tif*")]) < len(tp_index)) or (len([*yx_dir.glob("*.tif*")]) < len(tp_index)):

        try:
            existing_files_zy = zy_dir.glob("*.tif")
            existing_files_zy = [int(f.stem[-4:]) for f in existing_files_zy]
            
            existing_files_yx = yx_dir.glob("*.tif")
            existing_files_yx = [int(f.stem[-4:]) for f in existing_files_yx]
        except ValueError:
            raise IndexError("You likely only have one of the two final projection files either yx or zy. To progress, delete the single final projection file and click to run the max projection again.")
    
        tp_missing_yx = [t not in existing_files_yx for t in tp_index]
        tp_missing_zy = [t not in existing_files_zy for t in tp_index]
        
        if sum(tp_missing_yx) >= sum(tp_missing_zy):
            tp_index = tp_index[tp_missing_yx]
        elif sum(tp_missing_zy) > sum(tp_missing_yx):
            tp_index = tp_index[tp_missing_zy]
       
        print("Starting Z-projections for each timepoint in parallel...")

        with Pool(8) as p:
            p.starmap(project_slice,zip(repeat(stack),repeat(channel),repeat(yx_dir),repeat(zy_dir),tp_index))

        print("Z-projections finished")
        

        print("Projecting all Z-projections across T...")
        yx_max = tiff.imread([*yx_dir.glob("*.tif")])
        zy_max = tiff.imread([*zy_dir.glob("*.tif")])

        yx_max_final = np.max(yx_max,axis=0)
        tiff.imwrite(yx_dir.joinpath("yx_final_projection.tif"),yx_max_final)
        zy_max_final = np.max(zy_max,axis=0)
        tiff.imwrite(zy_dir.joinpath("zy_final_projection.tif"),zy_max_final)
        
        print("Time projections finished")
        
        return yx_max_final,zy_max_final,pixel_meta
    
    elif (len([*zy_dir.glob("*.tif*")]) > len(tp_index)) and (len([*yx_dir.glob("*.tif*")]) > len(tp_index)):
        print("All projections present, loading in data")
        yx_max_final = tiff.imread(yx_dir.joinpath("yx_final_projection.tif"))
        assert(yx_dir.joinpath("yx_final_projection.tif").exists()), "there is no file named yx_final_projection.tif"
        zy_max_final = tiff.imread(zy_dir.joinpath("zy_final_projection.tif"))
        assert(zy_dir.joinpath("zy_final_projection.tif").exists()), "there is no file named zy_final_projection.tif"

        return yx_max_final, zy_max_final,pixel_meta
    
    elif (len([*zy_dir.glob("*.tif*")]) == len(tp_index)) and (len([*yx_dir.glob("*.tif*")]) == len(tp_index)):
        print("Projecting all Z-projections across T...")
        yx_max = tiff.imread([*yx_dir.glob("*.tif")])
        zy_max = tiff.imread([*zy_dir.glob("*.tif")])

        yx_max_final = np.max(yx_max,axis=0)
        tiff.imwrite(yx_dir.joinpath("yx_final_projection.tif"),yx_max_final)
        zy_max_final = np.max(zy_max,axis=0)
        tiff.imwrite(zy_dir.joinpath("zy_final_projection.tif"),zy_max_final)
        print("Time projections finished")

        return yx_max_final,zy_max_final,pixel_meta
    else:
        raise IndexError("You likely only have one of the two final projection files either yx or zy. To progress, delete the single final projection file and click to run the max projection again.")
    

#this function is used in the cropping incase the cropped boundary is below zero, adjusts the boundary to zero 
def boundary_zero(dim_list):
    if any(np.array(dim_list) < 0):
        idx = np.where(np.array(dim_list)<0)[0]
        for i in idx:
            dim_list[i] = 0

    return dim_list

class settings_xml:

    def __init__(self,root=None):

        self._root = root
        
    
    def _xml_indent(self,elem, level=0):
            ###from npy2bdv
            """Pretty printing function"""
            i = "\n" + level * "  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    self._xml_indent(elem, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

            return elem

    def settings2xml(self,input_path):

        settings_dir = Path.joinpath(input_path.parent,"Settings")

        settings = []
        settings_name = []
        for f in settings_dir.glob("*"):
            with open(f,'r') as d:
                settings.append(json.load(d))
                settings_name.append(f.name)


        root = ET.Element('ViventisLS2_Settings')
        for i in range(len(settings)):
            elem = ET.SubElement(root,settings_name[i])
            if type(settings[i]) == dict:
                for key,val in settings[i].items():
                    child = ET.SubElement(elem,key).text = str(val)
            else:
                for j in range(len(settings[i])):
                    sub_elem = ET.SubElement(elem,"SettingPreset_"+str(j))
                    for key,val in settings[i][j].items():
                        child = ET.SubElement(sub_elem,key).text = str(val)
                    
        # settings_xml = ET.ElementTree(root)
        self._root = root
        self._xml_indent(self._root)
        
        return root


### this version of tiff conversion makes the server stored copy first and then the working copy second

def tiff2bdv(input_path:Path,input_key:str,output_path:Path,tstart:int,tend:int,crsid:str,compression_level = 2,stor_location = Path("\\\\gen-nas-pc-002\\Room-225\\Room225_SharedFolder\\ViventisLS2_data"),bit_depth = 'int16',pyramid_levels = ((1, 4, 4),    (2, 8, 8),    (4, 16, 16)),chunk_dims =((4, 250, 250),(4, 250, 250), (4, 250, 250), (4, 250, 250)), overwrite=False): #axes,chunk_dims, ,
    # TODO: save compression type and level into metadata
    start_time = time.time()
    new_filename = ("_").join(str(input_path.parent.name).split('_')[:3] + "".join(input_path.name.split(" ")).split('_')[:2]) #updated to remove spaces from viventis folder names
    output_path = output_path.joinpath(new_filename)
    
    
    # log_path = output_path.with_suffix(".log")
    # print(f"\n\nFOR INFO ON THIS CONVERSION THE LOG CAN BE FOUND AT: {log_path}")
    # logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG,format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    logger.info(f"Script Start {start_time}")

    output_stor = stor_location.joinpath(*output_path.parts[2:])  #\\\\gen-nas-pc-001.gen.private.cam.ac.uk\\lightsheet-nas\\
    output_stor.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created file tree on server at {output_stor.parent}")
    
    if overwrite == False:
        #logger.error("This h5 file already exists please remove it from the output folder and try again")
        assert (output_stor.with_suffix(".h5").exists() == False), "\n\nThis h5 file already exists please remove it from the output folder and try again"

    
    logger.debug(output_path)


    dask_stack, pixel_metadata,all_metadata = lazy_loading(input_path,input_key,tstart=tstart,tend=tend)
    x_dims, y_dims, z_dims = read_crop_dims(input_path,pixel_metadata)
    logger.info("Loaded data as virtual stack")

    if bit_depth == "int16":
        bdv_writer = npy2bdv.BdvWriter(str(output_stor.with_suffix(".h5")), nchannels=dask_stack.shape[1], nilluminations=1,
                                    nangles=1,overwrite=True,compression = hdf5plugin.Blosc(cname='zstd',clevel=compression_level,shuffle=1),bit_depth = bit_depth)
    elif bit_depth == "int8":
        bdv_writer = npy2bdv.BdvWriter(str(output_stor.with_suffix(".h5")), nchannels=dask_stack.shape[1], nilluminations=1,
                                    nangles=1,overwrite=True,compression = hdf5plugin.Blosc(cname='zstd',clevel=compression_level,shuffle=1),bit_depth = "u"+bit_depth)

    
    channel_names = []
    for chan in all_metadata[0][0]:
        if chan.tag.split("}")[-1] == 'Channel':
            chan_attr = chan.attrib
            channel_names.append(chan_attr['Name'])

    bdv_writer.set_attribute_labels('channel', tuple(channel_names))
    bdv_writer.set_attribute_labels('illumination', ('Fused',))
    bdv_writer.set_attribute_labels('angle', ('0',))
    
    bdv_writer.timestep=pixel_metadata["TimeIncrement"]
    bdv_writer.time_unit = pixel_metadata["TimeIncrementUnit"]
    
    logger.info(f"Beginning conversion of Storage Data with shape ({dask_stack.shape}),(t,c,x,y,z)")
    
    for t in range(dask_stack.shape[0]):
        for i_ch in range(dask_stack.shape[1]):
            for i_illum in range(1):
                for i_angle in range(1):
                    try:
                        logger.debug(f"Converting timepoint {t}, channel {i_ch}")
                        print(f"Converting timepoint {t}, channel {i_ch}")
                        if bit_depth == 'int16':
                            bdv_writer.append_view(dask_stack[t,i_ch,int(z_dims['z_start']):int(z_dims['z_end']),int(y_dims['y_start']):int(y_dims['y_end']),int(x_dims['x_start']):int(x_dims['x_end'])], time=t, channel=i_ch,
                                            illumination=i_illum, angle=i_angle,voxel_size_xyz=(pixel_metadata['PhysicalSizeX'],pixel_metadata['PhysicalSizeY'],pixel_metadata['PhysicalSizeZ']),
                                            voxel_units="micron", calibration=(1,1,float(pixel_metadata['PhysicalSizeZ'])/float(pixel_metadata['PhysicalSizeX'])))
                            bdv_writer._file_object_h5.flush()
                        elif bit_depth == 'int8':
                            img = dask_stack[t,i_ch,int(z_dims['z_start']):int(z_dims['z_end']),int(y_dims['y_start']):int(y_dims['y_end']),int(x_dims['x_start']):int(x_dims['x_end'])].compute()
                            vmin, vmax = np.percentile(img, q=(0, 95))
                            img_clip = rescale_intensity(img, in_range=(vmin, vmax), out_range=np.uint16)
                            bdv_writer.append_view(img_clip.astype(np.uint8), time=t, channel=i_ch,
                                            illumination=i_illum, angle=i_angle,voxel_size_xyz=(pixel_metadata['PhysicalSizeX'],pixel_metadata['PhysicalSizeY'],pixel_metadata['PhysicalSizeZ']),
                                            voxel_units="micron", calibration=(1,1,float(pixel_metadata['PhysicalSizeZ'])/float(pixel_metadata['PhysicalSizeX'])))
                            bdv_writer._file_object_h5.flush()
                    except:
                        logger.exception('')

    
    
    time_lag = time.time() - start_time
    logger.info(f"Storage data writing and compressing time, total: {time_lag/60:2.2f} min.")
    logger.info(f"Storage data size: {output_stor.with_suffix('.h5').stat().st_size*1e-9:2.2f} GB.")
    logger.info(f"Storage data conversion rate: {output_stor.with_suffix('.h5').stat().st_size/time_lag:2.2f} bytes/sec.")
    #logger.info("Now creating multiresolution pyramid")
    #start_time = time.time()
    #bdv_writer.create_pyramids(subsamp=pyramid_levels,blockdim=chunk_dims) #hdf5plugin.Blosc(cname='zstd',clevel=2,shuffle=1)
    #subsamp=pyramid_levels, 
                        #    blockdim=chunk_dims, 
    #logger.info(f"Pyramid writing and compressing time, total: {(time.time() - start_time)/60:2.2f} min.")
    logger.info("Now writing metadata to xml file")#print(bdv_writer.compression)
    
    bdv_writer.write_xml(microscope_name="viventisLS2",microscope_version="2.0.0.5", user_name=crsid)

    #bdv_writer.close()

    bdv_xml = ET.parse(output_stor.with_suffix(".xml"),parser = ET.XMLParser(encoding="utf-8"))
    xml_root = bdv_xml.getroot()
    xml_root.extend(all_metadata)

    settings = settings_xml().settings2xml(input_path)
    xml_root.extend(settings)

    complete_meta = settings_xml(xml_root)
    complete_meta._xml_indent(xml_root)
    
    bdv_xml = ET.ElementTree(complete_meta._root)
    bdv_xml.write(output_stor.with_suffix(".xml"),xml_declaration=True, encoding='utf-8', method="xml")
    
    bdv_writer.close()

    output_stor.with_suffix(".h5").rename(output_stor.with_suffix(".STOR.h5"))
    output_stor.with_suffix(".xml").rename(output_stor.with_suffix(".STOR.xml"))

    logger.info(f"Storage data saved at {output_stor.with_suffix('.STOR.h5')}")
    print(f"Storage data saved at {output_stor.with_suffix('.STOR.h5')}")

    logger.info("Duplicating data for active use")
    #tiff2storage(string_path,compression=hdf5plugin.Blosc(cname='zstd',clevel=2,shuffle=1))
    storage2bdv(output_path,bit_depth,stor_location,pyramid_levels,chunk_dims)

    # logger.info("Finished")
    # logging.shutdown()
    
    return

def storage2bdv(input_path:Path,bit_depth='int16',stor_location=Path("\\\\gen-nas-pc-002\\Room-225\\Room225_SharedFolder\\ViventisLS2_data"),pyramid_levels = ((1, 4, 4),    (2, 8, 8),    (4, 16, 16)),chunk_dims =((4, 250, 250),(4, 250, 250), (4, 250, 250), (4, 250, 250))):
    start_time = time.time()
    print("Beginning conversion of the Active data")
    input_path_stor = stor_location.joinpath(*input_path.parts[2:])
    output_path = input_path
    # stem = Path(input_path.stem).stem
    # output_path = destination_drive.joinpath(*input_path.parts[1:-1]).joinpath(stem)
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stor_file = h5py.File(input_path_stor.with_suffix(".STOR.h5"),"r") #with h5py.File(input_path,"r") as stor_file:

        channels = []
        timepoints = []
        for key in stor_file.keys():
            if "s" in key:
                channels.append(int(key[1:]))
                
            elif "t" in key:
                timepoints.append(int(key[1:]))
        logger.info(f"this file contains {len(channels)} channels and {len(timepoints)} timepoints")

        # with npy2bdv.BdvWriter(str(output_path)+".h5", nchannels=len(channels), nilluminations=1,
        #                             nangles=1,overwrite=False,compression = "gzip",bit_depth=bit_depth) as bdv_writer:
        bdv_writer = npy2bdv.BdvWriter(str(output_path)+".h5", nchannels=len(channels), nilluminations=1,
                                    nangles=1,overwrite=False,compression = "gzip",bit_depth=bit_depth)

        for t in range(len(timepoints)):
            for i_ch in range(len(channels)):
                for i_illum in range(1):
                    for i_angle in range(1):
                        logger.debug(f"Converting timepoint {t}, channel {i_ch}")
                        print(f"Converting timepoint {t}, channel {i_ch}")
                        stack = "t"+str(t).zfill(5)+"/s"+str(i_ch).zfill(2)+"/0/cells"
                        #bdv_writer.bit_depth = str(stor_file[stack].dtype)
            
                        bdv_writer.append_view(stor_file[stack], time=t, channel=i_ch,
                                            illumination=i_illum, angle=i_angle)
                        bdv_writer._file_object_h5.flush()

        
        stor_file.close()
        
        time_lag = time.time() - start_time
        logger.info(f"Active data writing and compressing time, total: {time_lag/60:2.2f} min.")
        logger.info(f"Active data size: {output_path.with_suffix('.h5').stat().st_size*1e-9:2.2f} GB.")
        raw_data_size = output_path.with_suffix(".h5").stat().st_size
        logger.info(f"Active data conversion rate: {output_path.with_suffix('.h5').stat().st_size/time_lag:2.2f} bytes/sec.")
        
        start_time = time.time()

        
        bdv_writer.create_pyramids(subsamp=pyramid_levels,blockdim=chunk_dims)
        #bdv_writer.close()

        time_lag = time.time() - start_time
        logger.info(f"Pyramid writing and compressing time, total: {time_lag/60:2.2f} min.")
        logger.info(f"Pyramid size: {(output_path.with_suffix('.h5').stat().st_size - raw_data_size)*1e-9:2.2f} GB.")
        logger.info(f"Pyramid conversion rate: {(output_path.with_suffix('.h5').stat().st_size - raw_data_size)/time_lag:2.2f} bytes/sec.")
        
    except:
        logger.exception('')

       
    
    logger.debug("copying xml file")
    shutil.copy(input_path_stor.with_suffix(".STOR.xml"), output_path.with_suffix(".xml"))
    logger.info(f"Active data can now be found at {output_path.with_suffix('.h5')}")
    print(f"Active data can now be found at {output_path.with_suffix('.h5')}") 
    
    return

def experiment2bdv(input_directory:Path,input_key:str,output_directory:Path,tstart:int,tend:int,crsid:str,compression_level = 2,bit_depth = 'int16',stor_location = Path("\\\\gen-nas-pc-002\\Room-225\\Room225_SharedFolder\\ViventisLS2_data"),pyramid_levels = ((1, 4, 4),    (2, 8, 8),    (4, 16, 16)),chunk_dims =((4, 250, 250),(4, 250, 250), (4, 250, 250), (4, 250, 250)), overwrite=False):
    # Guillermo notes for future users
    print('IMPORTANT: THE input_directory IS THE MAIN FOLDER EXPERIMENT CONTAINING THE MAXIMUM PROJECTION FOLDER AND THE SETTINGS FOLDER!')
    print('IMPORTANT: Your converted files will appear in this same folder when the conversion in completed')

    folders = [folder for folder in input_directory.iterdir() if folder.is_dir() and folder.stem != "Settings" and "max" not in folder.stem and folder.stem != "piv_ouput"]
    for position in folders:
        #input_directory = input_directory.joinpath(position)
        if all(["Transmitted" in fname.stem for fname in position.iterdir()][1:]):
            continue
        else:
            
            new_filename = ("_").join(str(position.parent.name).split('_')[:3] + "".join(position.name.split(" ")).split('_')[:2]) #updated to remove spaces from viventis folder names
            output_path = output_directory.joinpath(new_filename)
            output_stor = stor_location.joinpath(*output_path.parts[2:])

            log_path = output_path.with_suffix(".log")
            print(f"\n\nFOR INFO ON THIS CONVERSION THE LOG CAN BE FOUND AT: {log_path}")
            logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG,format="%(asctime)-15s %(levelname)-8s %(message)s")

            print(f"Started conversion of {position.stem}")
            logger.info(f"Started conversion of {position.stem}")
            if output_stor.with_suffix(".STOR.h5").is_file():
                print('Stored version of this file exists, creating active version...')
                storage2bdv(output_path,bit_depth=bit_depth,stor_location=stor_location,pyramid_levels=pyramid_levels,chunk_dims=chunk_dims)
            elif output_path.with_suffix(".h5").is_file():
                print('Active version of this file exists, moving to next file...')
                continue
            else:
                print('File doesnt exist, creating stored copy and active copy now...')
                tiff2bdv(input_path=position,input_key=input_key,output_path=output_path.parent,tstart=tstart,tend=tend,crsid=crsid,
                        compression_level= compression_level,bit_depth = bit_depth,stor_location=stor_location,pyramid_levels = pyramid_levels,chunk_dims =chunk_dims,
                        overwrite=overwrite)
            print(f"Finished conversion of {position.stem}")
            logger.info(f"Finished conversion of {position.stem}")

    logger.info("Finished all positions")
    logging.shutdown()

    return



#experiment2bdv(Path('Z:\\Room224_SharedFolder\\Alice\\pescoid\\20250304_165246_expzacy0044_fused'),"*Fused*",Path('Z:\\Room224_SharedFolder\\Alice\\pescoid\\20250304_165246_expzacy0044_fused'),0,-1,'ddzs2',bit_depth="int8",stor_location=Path("Z:\\Room224_SharedFolder\\Alice\\pescoid\\20250304_165246_expzacy0044_fused"))
# experiment2bdv(Path("D:\\Users\\Dillan\\for_mounting_measure\\20231124_105016_Experiment"),"*561*",Path("D:\\Users\\Dillan\\for_mounting_measure\\20231124_105016_Experiment"),0,1,'help')
# output_path =Path("G:\\Imen\\RA\\Exp13-RA-h5")
# input_path = Path("G:\\Imen\\RA\\20240927_141009_Exp13-SBNhybrid-RA_position2_fused")
# experiment2bdv(input_path,'*Fused*',output_path,0,None,"il")


###################################################################################################################


# function for cropping in serial 
# def project_time_series(input_path,input_key,tp_step, channel):
#     #create a dask stack for the image data
#     stack, pixel_meta, meta = lazy_loading(input_path,input_key)
#     print("Image data lazy loaded")
    
#     #create the folders to save the projections. this is to ensure that time isnt wasted for large movies
#     yx_dir = input_path.joinpath("0_yx_projection_DELETEME") #note: they are numbered 0 and 1 so they will be at the top of the list of directory contents
#     yx_dir.mkdir(exist_ok=True)
#     zy_dir = input_path.joinpath("1_zy_projection_DELETEME")
#     zy_dir.mkdir(exist_ok=True)

#     existing_files_zy = zy_dir.glob("*.tif")
#     existing_files_zy = [*existing_files_zy]

#     existing_files_yx = yx_dir.glob("*.tif")
#     existing_files_yx = [*existing_files_yx]

#     tp_index = np.array(range(0,stack.shape[0],tp_step))
    
#     if (len(existing_files_yx) < len(tp_index)) or (len(existing_files_zy) < len(tp_index)):
#         tp_count = min([len(existing_files_yx),len(existing_files_zy)])
#         for i in range(tp_index[tp_count],stack.shape[0],tp_step):
#             print(f'Projecting timepoint {i}. This is the {tp_count} timepoint out of {tp_index.shape[0]}')
            
#             yx_slice = np.max(stack[i,channel,...].compute(),axis=0)
#             tiff.imwrite(yx_dir.joinpath(f"yx_max_t{str(i).zfill(4)}.tif"),yx_slice)
            
#             zy_slice = np.max(stack[i,channel,...].compute(),axis=2)
#             tiff.imwrite(zy_dir.joinpath(f"zy_max_t{str(i).zfill(4)}.tif"),zy_slice)
            
#             tp_count+= 1
        
#         yx_max = tiff.imread([*yx_dir.glob("*.tif")])
#         zy_max = tiff.imread([*zy_dir.glob("*.tif")])

#         yx_max_final = np.max(yx_max,axis=0)
#         tiff.imwrite(yx_dir.joinpath("yx_final_projection.tif"),yx_max_final)
#         zy_max_final = np.max(zy_max,axis=0)
#         tiff.imwrite(zy_dir.joinpath("zy_final_projection.tif"),zy_max_final)

#         return yx_max_final,zy_max_final,pixel_meta

#     elif (len(existing_files_yx) > len(tp_index)) and (len(existing_files_zy) > len(tp_index)):
#         yx_max_final = tiff.imread(yx_dir.joinpath("yx_final_projection.tif"))
#         assert(yx_dir.joinpath("yx_final_projection.tif").exists()), "there is no file named yx_final_projection.tif"
#         zy_max_final = tiff.imread(zy_dir.joinpath("zy_final_projection.tif"))
#         assert(zy_dir.joinpath("zy_final_projection.tif").exists()), "there is no file named zy_final_projection.tif"

#         return yx_max_final, zy_max_final,pixel_meta
    
#     else:
#         yx_max = tiff.imread([*yx_dir.glob("*.tif")])
#         zy_max = tiff.imread([*zy_dir.glob("*.tif")])

#         yx_max_final = np.max(yx_max,axis=0)
#         tiff.imwrite(yx_dir.joinpath("yx_final_projection.tif"),yx_max_final)
#         zy_max_final = np.max(zy_max,axis=0)
#         tiff.imwrite(zy_dir.joinpath("zy_final_projection.tif"),zy_max_final)

#         return yx_max_final,zy_max_final,pixel_meta
    





## Functions for Zarr conversion if you want to switch to Zarr fro HDF5 ##

# from glob import glob
# from ome_zarr.writer import write_image
# import concurrent.futures as futures
# import zarr
# import numcodecs as num
# from ome_zarr.scale import Scaler
# from datetime import datetime
# from ome_zarr.io import parse_url
# from ome_zarr.format import CurrentFormat
# from ome_zarr.writer import _get_valid_axes,write_multiscales_metadata
# from skimage.io.collection import alphanumeric_key

# def insert_timepoint(i, zimg,dask_stack):
#     print(i)
#     zimg[i:i+1] = dask_stack[i:i+1].compute()

# def save_multiscale(scale_factor,dask_stack,root,compressor,pixel_metadata,chunk_dims):

#     if scale_factor != 0:
#         dask_stack = Scaler(downscale=scale_factor).resize_image(dask_stack)
    
#     if type(root) == h5py._hl.files.File:
#         try:
            
#             for tp in range(dask_stack.shape[0]):
#                 for sp in range(dask_stack.shape[1]):
#                     if scale_factor == 0:
#                         scale_id = 0
#                     else:
#                         scale_id = int(np.log2(scale_factor))

#                     name = "t"+str(tp).zfill(5)+"/s"+str(sp).zfill(2)+"/"+str(scale_id)+"/cells"
#                     root.create_dataset(name, data=dask_stack,dtype="int16", chunks = chunk_dims,compression = compressor)
#         except ValueError:
#             print(f'Skipped scale 0{scale_factor}, it already exists')
    
#     else:
#         name = str(scale_factor)
#         try:
#             zimg = root.zeros(name,shape = dask_stack.shape,compressor=compressor,dtype = pixel_metadata['Type'],chunks=chunk_dims) #,synchronizer=zarr.ThreadSynchronizer()
#             for i in range(len(dask_stack)):
#                 zimg[i:i+1] = dask_stack[i:i+1].compute()
#         except zarr.errors.ContainsArrayError:
#             print(f'Skipped scale 0{name}, it already exists')


# def multiscale_metadata(root,output_path,axes):
#     shapes = [root[i].shape for i in range(len(root))]
#     coord_transformations = CurrentFormat().generate_coordinate_transformations(shapes)
#     paths = [root[i].path for i in range(len(root))]
#     datasets = [{"coordinateTransformations":t,"path":p} for t,p in zip(coord_transformations,paths)]

#     val_axes = _get_valid_axes(len(root[0].shape),axes)

#     write_multiscales_metadata(group=root, datasets=datasets, fmt=CurrentFormat(), axes=val_axes, name = str(output_path).split('\\')[-1])
#     print("Multiscales metadata written")



# def tiff2omezarr(input_path,input_key,output_path,tstart,tend,axes,chunk_dims =(1,100,100,100) ):

#     ## IMPORTANT: since adding the HDF5 converter and allowing for multiple channels this function may need updating to account for different axis order etc

#     start_time = time.time()

#     dask_stack, pixel_metadata,all_metadata = lazy_loading(input_path,input_key,tstart=tstart,tend=tend)
#     print("Loaded data as dask stack")
#     compressor = num.blosc.Blosc(cname='zstd',clevel=2,shuffle=1)#num.zstd.Zstd(level=2)

#     #assert(os.path.isdir(output_path) == False), "OME Zarr is already present at output path"
#     if not output_path.is_dir():#os.path.isdir(output_path):
#         output_path.mkdir()

#     print("Created zarr directory")
#     store = parse_url(output_path, mode="w").store
#     root = zarr.group(store=store)
#     print("Created Zarr Group")
#     # executor = futures.ProcessPoolExecutor(10)
#     print("Saving multiscales level 0...")
#     save_multiscale(0,dask_stack,root,compressor,pixel_metadata,chunk_dims)
#     print("Saving multiscales level 1...")
#     save_multiscale(2,dask_stack,root,compressor,pixel_metadata,chunk_dims)
#     print("Saving multiscales level 2...")
#     save_multiscale(4,dask_stack,root,compressor,pixel_metadata,chunk_dims)
#     print("Saving multiscales level 3...")
#     save_multiscale(8,dask_stack,root,compressor,pixel_metadata,chunk_dims)
#     print("Saving multiscales level 4...")
#     save_multiscale(16,dask_stack,root,compressor,pixel_metadata,chunk_dims)

#     print("Writing Metadata to zarr attrs")
#     multiscale_metadata(root,output_path,axes)
#     root.attrs["OME-Zarr_Conversion"] = dict(
#         creation_time = datetime.now().isoformat(),
#         conversion_time = str((time.time() - start_time)/3600) + " hrs",
#         compression = str(compressor),
#         chunking = root[0].chunks
#     )
#     print("Zarr conversion metadata written")
#     root.attrs["Acquisition_Metadata"] = all_metadata
#     settings_folder = sorted(input_path.parent.glob("Settings"))[0]
#     if settings_folder.is_dir():
#         for file in settings_folder.glob("*") :
#             #new_file = output_path/file.name
#             new_paths = shutil.copy(file,output_path)
#     else:
#         print("Could not copy microscope settings to Zarr folder")
#     print("Acquistion metadata written")

#     verify_compression = root[0][0].shape
#     print("Done")

#     return 






