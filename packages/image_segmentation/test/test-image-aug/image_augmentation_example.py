# Don't forget to do this before running this Python script:
# module load python/3.6
#And add imgaug to the PYTHONPATH

# Inputs
composite_output_dir = 'composite_aug'
individual_output_dir = 'individual_aug'

# Import general Python modules
import numpy as np
import sys, os, time

import image_segmentation.utils as imseg_utils # import the utils module
from image_segmentation.image_augmentation import augment_images # import the augment_images function in the image_augmentation module

# Load the data
sample_dir = "../../sample_data/" 
lady_rgb_image = os.path.join(sample_dir, "lady_images_rgb_original_15.npy")
lady_mask = os.path.join(sample_dir, "lady_masks_original_15.npy")

# Create the stacks of sizes neach*2
images = np.load(lady_rgb_image)
masks = np.load(lady_mask)

# Do the composite augmentations, returning the augmented images and masks as numpy arrays
# Since output_dir is set, we also output .tif files for examination
# num_aug=1 because we manually did the repetitions in our case in order to combine two regions and settle on neach*2 total stack size
# aug_params and composite_sequence will use default values since they are set to None. To customize these, take as examples the settings in fnlcr-bids-hpc/image_segmentation/image_augmentation.py
if not os.path.isdir(composite_output_dir):
    os.mkdir(composite_output_dir)

time_start = time.time()
images_aug, masks_aug = augment_images(images, masks=masks, num_aug=1, output_dir=composite_output_dir)
time_end = time.time()
print('Composite augmentation took {:4.1f} min'.format((time_end-time_start)/60))

# Do the individual augmentations, returning the list of augmented images as numpy arrays
# Since output_dir is set, we also output .tif files for examination
# num_aug=1 because we manually did the repetitions in our case in order to combine two regions and settle on neach*2 total stack size
# aug_params and individual_seqs_and_outnames will use default values since they are set to None. To customize these, take as examples the settings in fnlcr-bids-hpc/image_segmentation/image_augmentation.py
if not os.path.isdir(individual_output_dir):
    os.mkdir(individual_output_dir)
time_start = time.time()
images_aug_indiv_list = augment_images(images, num_aug=1, do_composite=False, output_dir=individual_output_dir)
time_end = time.time()
print('Individual augmentations took {:4.1f} min total'.format((time_end-time_start)/60))

# Output information about the input and output numpy arrays
imseg_utils.arr_info(masks_aug)
