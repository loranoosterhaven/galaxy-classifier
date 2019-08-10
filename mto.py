import mtolib.main as mto

import numpy as np

"""Example program - using original settings"""

# Get the input image, parameters and coordinates
image, params, coords = mto.setup()

# Pre-process the image
processed_image = mto.preprocess_image(image, params, n=2)

# Build a max tree
mt = mto.build_max_tree(processed_image, params)

# Filter the tree and find objects
id_map = mto.filter_tree(mt, processed_image, params)

# obtain nodes array from MaxTree object
nodes = np.ctypeslib.as_array(mt.nodes, shape=image.shape) 
node_attribs = np.ctypeslib.as_array(mt.node_attributes, shape=image.shape) 

# Relabel objects for clearer visualisation
id_map = mto.relabel_segments(id_map, shuffle_labels=False)

# open catalogue if requested
gz_cat = mto.read_gz_catalogue(params.gz_filename)

# Generate output files
mto.generate_image(image, id_map, params)
mto.generate_parameters(image, gz_cat, id_map, nodes, node_attribs, coords, params)