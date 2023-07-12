Input
************

The input for the :func:`undouble.undouble.Undouble.import_data` can be the following three types:

    * Directory path
    * File locations
    * Numpy array containing images

The scanned files and directories can also be filtered on extention type, or directories can be black listed. Note that these settings need to be set during initialization. The black_list directory is set to undouble by default to make sure that readily moved files are not incorporated in the analysis.

The following parameters can be changed during initialization:

    * Images are imported with the extention ([‘png’,’tiff’,’jpg’,’jfif’]).
    * Input image can be grayscaled during import.
    * Resizing images to save memory, such as to (128, 128).



Directory
======================

Images can imported recursively from a target directory.

.. code:: python

	# Import library
	from undouble import Undouble

	# Init with default settings
	model = Undouble()

	# Import data
	input_list_of_files = model.import_example(data='flowers')
	input_directory, _ = os.path.split(input_list_of_files[0])

	# The target directory looks as following:
	print(input_directory)
	# 'C:\\TEMP\\flower_images'

	# Importing the files files from disk, cleaning and pre-processing
	model.import_data(input_directory)

	# [clustimage] >INFO> Extracting images from: [C:\\TEMP\\flower_images]
	# [clustimage] >INFO> [214] files are collected recursively from path: [C:\\TEMP\\flower_images]
	# [clustimage] >INFO> [214] images are extracted.
	# [clustimage] >INFO> Reading and checking images.
	# [clustimage] >INFO> Reading and checking images.
	# [clustimage]: 100%|██████████| 214/214 [00:01<00:00, 133.25it/s]

	# Compute hash
	model.compute_hash()
    
	# Find images with image-hash <= threshold
	model.group(threshold=0)

	# Plot the images
	model.plot()


File locations
======================

Read images recursively from a target directory.

.. code:: python

	# Import library
	from undouble import Undouble

	# Init with default settings
	model = Undouble()

	# Import data; Pathnames to the images.
	input_list_of_files = model.import_example(data='flowers')

	# [undouble] >INFO> Store examples at [..\undouble\data]..
	# [undouble] >INFO> Downloading [flowers] dataset from github source..
	# [undouble] >INFO> Extracting files..
	# [undouble] >INFO> [214] files are collected recursively from path: [..\undouble\undouble\data\flower_images]
	
	# The list image path locations looks as following but may differ on your machine.
	print(input_list_of_files)

	# ['\\repos\\undouble\\undouble\\data\\flower_images\\0001.png',
	#  '\\repos\\undouble\\undouble\\data\\flower_images\\0002.png',
	#  '\\repos\\undouble\\undouble\\data\\flower_images\\0003.png',
	#  ...]

	model.import_data(input_list_of_files)

	# [clustimage] >INFO> Reading and checking images.
	# [clustimage] >INFO> Reading and checking images.
	# [clustimage]: 100%|██████████| 214/214 [00:02<00:00, 76.44it/s]

	# Compute hash
	model.compute_hash()
    
	# Find images with image-hash <= threshold
	model.group(threshold=0)

	# Plot the images
	model.plot()



Numpy Array
======================

Images can also be in the form of a numpy-array.

.. code:: python

	# Import library
	from undouble import Undouble

	# Init with default settings
	model = Undouble()

	# Import data; numpy array containing images.
	X, y = model.import_example(data='mnist')

	print(X)
	# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
	#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
	#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
	#        ...,
	#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
	#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
	#        [ 0.,  0., 10., ..., 12.,  1.,  0.]])

	# Compute hash
	model.compute_hash()
    
	# Find images with image-hash <= threshold
	model.group(threshold=0)

	# Plot the images
	model.plot()



Output
************

The output is stored in model.results

.. code:: python

	# Import library
	from undouble import Undouble

	# Print all keys
	print(model.results.keys())

	# dict_keys(['img',
	#            'pathnames',
	#            'url',
	#            'filenames',
	#            'img_hash_bin',
	#            'img_hash_hex',
	#            'adjmat',
	#            'select_pathnames',
	#            'select_scores',
	#            'select_idx',
	#            'stats'])

	# Pathnames
	model.results['pathnames']

	# array(['D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0001.png',
	#        'D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0002.png',
	#        'D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0003.png',...
	
	# Filenames
	model.results['filenames']
	# array(['0001.png', '0002.png', '0003.png',...
	
	# Adjacency matrix
	model.results['adjmat']
	# array([[ 0, 24, 24, ..., 30, 28, 26],
	#        [24,  0, 28, ..., 28, 18, 36],
	#        [24, 28,  0, ..., 28, 28, 28],
	#        ...,
	#        [30, 28, 28, ...,  0, 24, 34],
	#        [28, 18, 28, ..., 24,  0, 34],
	#        [26, 36, 28, ..., 34, 34,  0]])
	
	# Select groupings
	model.results['select_idx']
	# [array([81, 82], dtype=int64),
	#  array([90, 91, 92], dtype=int64),
	#  array([169, 170], dtype=int64)]


Extract Groups
******************

Extracting the groups can be done using the group-index combined with the pathnames (or filenames).

.. code:: python

	# Import library
	from undouble import Undouble

	# Init with default settings
	model = Undouble()

	# Import data; Pathnames to the images.
	input_list_of_files = model.import_example(data='flowers')

	# Import data from files.
	model.import_data(input_list_of_files)

	# Compute hash
	model.compute_hash()
    
	# Find images with image-hash <= threshold
	model.group(threshold=0)

	# [undouble] >INFO> [3] groups with similar image-hash.
	# [undouble] >INFO> [3] groups are detected for [7] images.

	# Plot the images
	model.plot()

	# Extract the pathnames for each group
	for idx_group in model.results['select_idx']:
	    print(idx_group)
	    print(model.results['pathnames'][idx_group])


	# [81 82]
	# ['D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0082 - Copy.png'
	#  'D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0082.png']
	# [90 91 92]
	# ['D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0090 - Copy (2).png'
	#  'D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0090 - Copy.png'
	#  'D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0090.png']
	# [169 170]
	# ['D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0167 - Copy.png'
	#  'D:\\REPOS\\undouble\\undouble\\data\\flower_images\\0167.png']



.. include:: add_bottom.add