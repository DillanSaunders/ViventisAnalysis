module batch_quickpiv

using quickPIV # for PIV
using LIBTIFF # for tiff loading
using DelimitedFiles #for csv handling
using HDF5 # for hdf5 handling


""" 
	searchdir(input_path::String, input_key::String)

Return the complete paths of all files in folder `input_path` where the file name contains `input_key`.
"""
function searchdir(input_path::String, input_key::String)
	
	files = filter(x->contains(x,input_key),readdir(input_path))

	full_path = Vector()
	for f in 1:length(files)
		append!(full_path,[string(input_path,files[f])])
	end

	return full_path
end


"""
	batch_piv(input_path::String,input_key::String,is_h5::Bool,target_channel::Int,window_size=40,search=20,voxel_dims=(0.406,0.406,2))

Run quickPIV's 3D PIV on all timepoints of a dataset.

Note: from the perspecitve of viventis LS2 output this function processes one position at a time regardless of whether the data is in .tiff or .h5 format. 
It must be run again for every position within an experiment. 

Arguments:
	`input_path`: path to folder containing the image timeseries
	`input_key`: used to select which files will be processed. For tiff files this can be used to select the channel of interest. For H5 files this can be used to select the single file of interest.
	`is_h5`: true or false depending on the file type being analysed
	`target_channel`: image channel to be analysed when processing H5 files. Still base zero because H5 channel order is base zero.
	`window_size`: Window Size. the size of the window used for the PIV analysis in pixels. will be later adjusted for Z-anisotropy
	`search`: Search Margin. the size by which to increase the PIV_WindowSize when searching for cross-correllation in pixels
	`voxel_dims`: The dimensions of a single voxel in the image data. 

Returns none but saves PIV vectorfield as .vtk files in a new folder 'piv_ouput' within `input_path`.
Vectorfield gives displacement in Y, X, Z (U, V, W)
"""
function batch_piv(input_path::String,input_key::String,is_h5::Bool,target_channel::Int,window_size::Int,search::Int,voxel_dims::Tuple{Float64, Float64, Int64})
	#window_size=40,search=20,voxel_dims=(0.406,0.406,2)
	#alice window_size=40,search=20,voxel_dims=(0.406,0.406,2)
	#imen window_size=52,search=32,voxel_dims=(0.26,0.26,2)

	#get paths of all files of interest
	full_path = searchdir(input_path,input_key)

	split_path = split(full_path[1],"\\")
	base_name = split_path[lastindex(split_path)]
	base_name = base_name[1:lastindex(base_name)-3]

	#create directory for data output within the input directory if it doesnt yet exist
	dir_present = filter(x->contains(x,base_name*"_piv_output"),readdir(input_path))
	if sizeof(dir_present) == 0
		mkdir(string(input_path,base_name*"_piv_output"))
	end

	#the location to save the data to is then set to this piv_output directory. Will overwrite files if it exists.
	output_path = string(input_path,base_name*"_piv_output\\")

	
	# TO DO: move params out on its own, easier to set and change parameters
	# voxel_dims = (0.406,0.406,2) # TO DO read voxel dims from ome.companion file

	#create an instance of the quickPIV parameter class with specified parameters
	params = quickPIV.setPIVParameters( interSize=window_size, searchMargin=search,overlap=0, corr="nsqecc",threshold = 300)

	#quickpiv does not take in to account z-anisotropy in its default state so we edit the parameter class and change the chunk size tuple
	args_list = collect(params.args)
	args_list[1] = (params.interSize[1],params.interSize[2],Int(round((params.interSize[1] * voxel_dims[1])/voxel_dims[3])))
	args_list[2] = (params.searchMargin[1],params.searchMargin[2],Int(round((params.searchMargin[1] * voxel_dims[1])/voxel_dims[3])))
	args_list[3] = (params.overlap[1],params.overlap[2],Int(round((params.overlap[1] * voxel_dims[1])/voxel_dims[3])))
	params.args = Tuple(args_list)
	params.interSize = (params.interSize[1],params.interSize[2],Int(round((params.interSize[1] * voxel_dims[1])/voxel_dims[3])))
	params.searchMargin = (params.searchMargin[1],params.searchMargin[2],Int(round((params.searchMargin[1] * voxel_dims[1])/voxel_dims[3])))
	params.overlap = (params.overlap[1],params.overlap[2],Int(round((params.overlap[1] * voxel_dims[1])/voxel_dims[3])))

	
	
	# if input file is HDF5 then open the file as an fileoject and set the number of iterations for the PIV to the number of timepoints in the H5 file
	if is_h5
		file = h5open(full_path[1],"r")
		num_itrs = parse(Int,keys(file)[length(keys(file))][2:6])#removed plus 1 to stop error at end
	# if input files are tiff then the list of full_paths will give you the number of iterations for the PIV
	else
		num_itrs = length(full_path)-1
	end
	
	#loop over each iteration. multithreaded so iterations begin in parallel.
	
	Threads.@threads for f in 1:num_itrs # 
		println("Loading Images...")
		# for every iteration take an image and the image after it
		if is_h5
			volume1 = Float32.(permutedims(read(file[string("t",lpad(f-1,5,"0"))][string("s",lpad(target_channel,2,"0"))]["0"]["cells"]),[2,1,3])) #the h5 package in julia opens in true col maj order, XYZ so for now permute dims
			volume2 = Float32.(permutedims(read(file[string("t",lpad(f,5,"0"))][string("s",lpad(target_channel,2,"0"))]["0"]["cells"]),[2,1,3]))
		else
			volume1 = LIBTIFF.tiffread(full_path[f],typ=Float32); #import note: the quickpiv implementation of libtiff loads images in Y,X,Z dimensions
			volume2 = LIBTIFF.tiffread(full_path[f+1],typ=Float32);
		end
		println("Starting PIV...")
		# calculate PIV for the two adjacent images using the parameters specified earlier
		U, V, W, SN = quickPIV.PIV( volume1, volume2, params ) # U is displacement in Y, V is disp in X, and W is disp in Z. SN gives the quality of the displacement estimation

		
		#take original image file names to label output vector field
		
		output_key = string("t",lpad(f,5,"0"),"_s",lpad(target_channel,2,"0"))
		

		#save vector field as .vtk file
		quickPIV.vectorFieldToVTK(output_key, U, V, W, path=output_path) #on output quickPIV switches dimensions so the disp array is V,U,W
		println("Vectors Saved")
	end
	
	
	println("Finished")
	
end

"""
	single_piv(f::Int,params,target_channel::Int,output_path::String,is_h5::Bool,full_path,file)

calculate PIV for a given timepoint number `f` and its consequtive timepoint.
"""
function single_piv(f::Int,params,target_channel::Int,output_path::String,is_h5::Bool,full_path,file)
	println("Loading Images...")
	if is_h5
		volume1 = Float32.(permutedims(read(file[string("t",lpad(f,5,"0"))][string("s",lpad(target_channel,2,"0"))]["0"]["cells"]),[2,1,3])) #the h5 package in julia opens in true col maj order, XYZ so for now permute dims
		volume2 = Float32.(permutedims(read(file[string("t",lpad(f+1,5,"0"))][string("s",lpad(target_channel,2,"0"))]["0"]["cells"]),[2,1,3]))
	else
		volume1 = LIBTIFF.tiffread(full_path[f],typ=Float32); #import note: the quickpiv implementation of libtiff loads images in Y,X,Z dimensions
		volume2 = LIBTIFF.tiffread(full_path[f+1],typ=Float32);
	end
	println("Starting PIV...")
	U, V, W, SN = quickPIV.PIV( volume1, volume2, params ) # U is displacement in Y, V is disp in X, and W is disp in Z

		# push!(Ut,U)
		# push!(Vt,V)
		# push!(Wt,W)
		# println(size(Ut))
		#take original image file names to label output vector field
		
	output_key = string("t",lpad(f,5,"0"),"_s",lpad(target_channel,2,"0"))
		
		# output_key = split(full_path[f],"\\")
		# output_key = String(split(output_key[length(output_key)],".")[1])

		#save vector field as .vtk file
	quickPIV.vectorFieldToVTK(output_key, U, V, W, path=output_path) #on output quickPIV switches dimensions so the disp array is V,U,W
	println("Vectors Saved")
end

""" 
	prepare_piv(input_path::String,input_key::String,is_h5::Bool,window_size=40,search=20)

Creates and returns PIV parameter object, piv output directory and list of files.
To be used with single_piv.
This splits up the batch_quickpiv function into two parts.
tested and not sure it improves anything
"""
function prepare_piv(input_path::String,input_key::String,is_h5::Bool,window_size=40,search=20)

	#create directory for data output within the input directory
	dir_present = filter(x->contains(x,"piv_output"),readdir(input_path))
	if sizeof(dir_present) == 0
		mkdir(string(input_path,"piv_output"))
	end
	output_path = string(input_path,"piv_output\\")

	full_path = searchdir(input_path,input_key)
	# TO DO: move params out on its own, easier to set and change parameters
	voxel_dims = (0.406,0.406,2) # TO DO read voxel dims from ome.companion file
	params = quickPIV.setPIVParameters( interSize=window_size, searchMargin=search,overlap=0, corr="nsqecc",threshold = 300)

	#quickpiv does not take in to account z-anisotropy in its default state so these lines go into the parameter class and change the chunk size tuple
	args_list = collect(params.args)
	args_list[1] = (params.interSize[1],params.interSize[2],Int(round((params.interSize[1] * voxel_dims[1])/voxel_dims[3])))
	args_list[2] = (params.searchMargin[1],params.searchMargin[2],Int(round((params.searchMargin[1] * voxel_dims[1])/voxel_dims[3])))
	args_list[3] = (params.overlap[1],params.overlap[2],Int(round((params.overlap[1] * voxel_dims[1])/voxel_dims[3])))
	params.args = Tuple(args_list)
	params.interSize = (params.interSize[1],params.interSize[2],Int(round((params.interSize[1] * voxel_dims[1])/voxel_dims[3])))
	params.searchMargin = (params.searchMargin[1],params.searchMargin[2],Int(round((params.searchMargin[1] * voxel_dims[1])/voxel_dims[3])))
	params.overlap = (params.overlap[1],params.overlap[2],Int(round((params.overlap[1] * voxel_dims[1])/voxel_dims[3])))


	if is_h5
		file = h5open(full_path[1],"r")
		num_itrs = parse(Int,keys(file)[length(keys(file))][2:6])
	else
		num_itrs = length(full_path)-1
		file = nothing
	end

	return params,file,full_path,output_path,num_itrs
end

"""
	 batchload_vectors(input_path::String,input_key::String;typ = nothing,tstart=1,tend=nothing)

Load in all vectors created by running batch_quickpiv located at `input_path` with filenames containing `input_key`. Between `tstart` and `tend`.
"""
function batchload_vectors(input_path::String,input_key::String;typ = nothing,tstart=1,tend=nothing)

	full_paths = searchdir(input_path,input_key)

	if tend==nothing
		full_paths = full_paths
	else
		full_paths = full_paths[tstart:tend]
	end
	
	Ut = Vector()
	Vt = Vector()
	Wt = Vector()
	for f in 1:(length(full_paths))
		filename = String(last(split(full_paths[f],"\\")))
		path = String(join(split(full_paths[f],"\\")[1:length(split(full_paths[f],"\\"))-1],"\\")) * "\\"

		U,V,W = quickPIV.loadVTKVectorField(filename,typ=typ,path=path)

		push!(Ut,U)
		push!(Vt,V)
		push!(Wt,W)
	end

	Ut = cat(Ut...,dims=4)
	Vt = cat(Vt...,dims=4)
	Wt = cat(Wt...,dims=4)
	
	return Ut,Vt,Wt;
end

"""
Not sure what i wrote this for
"""
function adder(x,y)
	return x+y
end

## Note: quickPIV does contain a function PIVTrajectories for generating pseudotracks from vector fields. I tried to implement this but could not get it to work so wrote my own version in quickpiv.py.




export batch_piv, batchload_vectors, single_piv, prepare_piv,adder

end # module batch_quickpiv


