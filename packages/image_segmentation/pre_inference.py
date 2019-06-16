# These are "compound" functions used in the pre-inference workflow. Simpler functions should be placed in the utils module.

def pre_process_inference_images(images_npy_file, idtype, nlayers_max, prefix, dir='.'):
    # Load the images, scale them, make the other two dimensions accessible, pad the resulting images, and save them to disk

    # Load relevant modules
    import numpy as np
    from . import utils

    # Load the images
    images = np.load(dir+'/'+images_npy_file)

    # Automatically scale the images
    images = utils.normalize_images(images,idtype)

    # Make the other two dimensions of the images accessible
    x_first,y_first,z_first = utils.transpose_stack(images)

    # Pad the each of 2D planes of the images
    x_first = utils.pad_images(x_first,nlayers_max)
    y_first = utils.pad_images(y_first,nlayers_max)
    z_first = utils.pad_images(z_first,nlayers_max)

    # Write these other "views" of the images to disk
    np.save(prefix+'-x_first.npy',x_first)
    np.save(prefix+'-y_first.npy',y_first)
    np.save(prefix+'-z_first.npy',z_first)