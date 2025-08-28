hi congratulations on logging on to HONEY BADGER!

you're probably excited to process and analyse some viventis lightsheet data right now

here's how to do it...

if this is your first time logging on to honey badger using your own profile you'll need to map our servers so you can access the data
This should only need to be done once.

	MAPPING THE DRIVES:
	Go to "This PC" in the file explorer
	Click "..." in the top bar
	Click "map network drive"

	Its ideal if the drives are in the same order for everyone so then code will work across profiles.
	please map the following servers to the associated drive letter
	
	Z: ->>> \\gen-nas-pc-001.gen.private.cam.ac.uk\Room-224
	Y: ->>> \\gen-nas-pc-002.gen.private.cam.ac.uk\Room-225
	X: ->>> \\larch.gen.private.cam.ac.uk\Steventon (this requires logon through DEPT rather than BLUE, will likely change in future)

	If needed input your BLUE login details and click remember

Your raw data:
> this should be located on the synology server in Room224 within a folder with your name
> the raw data should be have both microscope views fused, if your data is not yet fused, do that first AND DO IT LOCALLY ON VIVENTIS. 
> the top level folder produced by the viventis is referred to as the EXPERIMENT
> the experiment folder will contain one or more POSITIONS
> the positions folder('s) will contain individual tiff stacks for each timepoint and channel

STEP 1: CONVERT YOUR DATA TO HDF5

We have set up a specific work flow that creates HDF5/XML file pairs that can be viewed in FIJI/Imaris, stored efficiently, and contain all the necessary metadata from the viventis. 

To run the workflow open "Anaconda Prompt" on your desktop. 
Paste the following commands into the prompt and hit enter 

	conda activate hdf5_conversion_env

	python C:\Users\Public\data_processing_scripts\VIVENTIS_HDF5_CONVERSION.py

A napari window should appear with several custom widgets on the right-hand side:
 > Cropping Lightsheet Data
 > Convert Lightsheet Experiment
 > Retrieve STORed File

(Optional) CROP YOUR DATA
 * Enter the location of the POSTION folder you want to crop. Cropping must be done one position at a time but you don't need to restart napari in between. 

 * Enter the channel number that you want to use to crop the data eg channel 0, channel 1, channel 2. This should be the channel with the most uniform signal

 * Enter the timestep, this allows you to increase the speed by using less timepoints but if your data is moving a lot then you should use more timepoints and have a lower        timestep 

 * When ready click "Max Project Data". This will project all your data together (across the timestep used) and it will open it in the Napari window. this will allow you to determine where the maximum extent of your data is in X, Y, and Z. 

 * Two maximum projections will open YX and ZY. Each is accompanied by a Cropbox (red). For each projection move and resize the cropbox to select the region of the image you want cropped. When ready click the corresponding Capture Dimensions button. This will log the dimensions you've chosen. 

 * When both YX and ZY dimensions have been captured Click "Save Crop Dimensions" which will save a .xml file into the positions folder called "CropDimensions.xml". 

 * Open the folder that contains the position you're cropping in FileExplorer. You should now see two new folders that contain the projections. When you have your cropped dimensions saved you can DELETE these folders.

 * Once all the positions within an EXPERIMENT have had their CropDimensions saved you are then ready to run the File Conversion.

CONVERT LIGHTSHEET EXPERIMENT

 * Enter the location of the experiment you want to convert. 
The script will convert all positions in this experiment into individual H5 files each with an accompanying xml file that contains the metadata, unless the position is a maximum projection or they are transmitted light (single slice)
the resulting files will be saved within the experiment folder
REMEMBER: each .h5 file has an xml companion file - DO NOT SEPARATE THESE FILES OR CHANGE THEIR NAMES

 * Enter input_key if you want to subset your data, otherwise leave it as the default "*Fused*"
 * Change time start and end if you want to only convert a subset of your data otherwise leave as default

 * Enter your CRSID so we can identify who converted the file. 

 * Click convert

What happens to the data, in brief...

	(a) first we create a highly compressed copy of the data on the server in room 225. this will copy the same file structure from the server where your data is located to make it easier to find. Remember to delete this copy if you do not want your data anymore
	(b) next we create a less compressed copy of the data in the same experiment folder where your raw data lives. This will be the "active" copy for analysis. 
	(c) finally, we create a multi-resolution pyramid within the active copy of the data. This is what allows for dynamic viewing of large datasets in fiji

If the stored copy of the file already exists, the script will skip to step (b) 
if you are curious how long it takes, each experiment will also generate a log text file which saves the time taken for each step for each position


STEP 2: CHECK SUCCESSFUL CONVERSION IN FIJI AND DELETE YOUR TIFFS

to ensure the conversion has been successful open your files in fiji and visually inspect them. 

open the fiji shortcut on your desktop

Under "Plugins" click on "BigDataViewer" and then on "Open XML/HDF5"

A file window dialog will open, navigate to where you newly converted data is and click on the XML file that corresponds to the data you want to view

A BigDataViewer window should then open. Use the information located under the "help" tab or online to navigate through the data and confirm that the conversion has been successful 

If there are white chunks/boxes/lines the conversion did not work properly, delete the converted data and try again (don't forget the stored copy!)

Once the check has been successful DELETE ALL YOUR TIFF FILES UP TO THIS POINT. REMEMBER TO ALSO DELETE THEM FROM THE RECYCLE BIN. if really necessary we can regenerate them from the HDF5 file but they take up way too much space to leave lying around.

STEP 3: ANALYSE YOUR DATA

FIJI (tracking etc):
	open the fiji shortcut on your desktop
	
	this copy of fiji is open to everyone so its important we don't break anything
	If you need to install new packages ask first! 
	***DO NOT UPDATE FIJI***

	For tracking use Mastodon/ELEPHANT (both are installed, found under Plugins>Tracking>Mastodon) or Ultrack

Imaris: 
	To use imaris log on to Porcupine
	Imaris can open .h5 files but it is very slow for large files
	if your file is large convert your data to .ims format


Python (3D PIV):

open anaconda prompt on your desktop

if this is your first time running the PIV analysis on this computer then you first need to make sure the local Julia environment is created. Enter the following commands into the prompt:
	> julia
	> ] 
	> activate C:\Users\Public\data_processing_scripts\batch_quickpiv 
	> instantiate
	> backspace key
	> exit()
This only needs to be done once. Further detail can be found in the Viventis Analysis Computer Set up Document.

Paste the following commands into the prompt and hit enter 

	conda activate piv3d_env

	python C:\Users\Public\data_processing_scripts\VIVENTIS_3DPIV_ANALYSIS.py

A napari window will open. Note: only one Postion/Embryo can be analysed at one time. It will not analyse the whole experiment.

	Run 3D PIV:
	> Choose a directory: select experiment folder containing the position you want to analyse
	> Key: put in a fragment of the filename of the postion/embryo to analyse
	> hdf5 target channel: the channel that you want to run the PIV on. options = [0,1,2]

	Register and View PIV Vectorfields:
	> choose folder containing the PIV results (should contain the words piv_output)
	> put in the PIVWindow size you used for the analysis and the voxel dimensions
	> Click register to perform Kabsch registration. This will produce two new .vtk files that will be saved in the PIV output folder. They will be called "rigid" and "flow". Rigid is the global movement, and flow is the local movement. 
	> For viewing select the type of vector field you want to view
	> select the property you want to colour the vectorfield with 
	> click visualise vectors fields

	Create Pseudotracks
	> choose folder containing the PIV results (should contain the words piv_output)
	> put in the PIVWindow size you used for the analysis and the voxel dimensions
	> pstrk random: this determines whether the script seeds random points in the vector field to create tracks or whether it starts with the center of every PIV window (this can be slow and it will be many tracks)
	> pstrk points: the number of points that are randomly seeded to generate tracks if pstk random is ticked



STEP 4: DELETE HDF5 FILES

Once you have analysed your data delete the HDF5/XML pair that you have been working on. This is only the active copy and can always be restored from the stored copy. 

STEP 5: RESTORE ACTIVE DATA

Use the widget available in the HDF5 conversion napari window.

You need to select the file you want to convert back to the active HDF5. This should have a .STOR tag on the end of the filename. 


DO NOT DOWNLOAD ANY OTHER PROGRAMES OR UPDATES WITHOUT ASKING BEN FIRST.






