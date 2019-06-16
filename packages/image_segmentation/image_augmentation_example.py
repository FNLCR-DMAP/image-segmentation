# Don't forget to do this before running this Python script:
# module load python/3.6

# Inputs
nlayers = 6
neach = 2500
#fnlcr_bids_hpc_path = '/data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc'
fnlcr_bids_hpc_path = '/home/weismanal/checkouts/fnlcr-bids-hpc'
imgaug_path = '/data/BIDS-HPC/public/software/checkouts/imgaug'
# Shapes of the following four numpy arrays are (N,H,W)
roi1_images_npy_file = '/data/BIDS-HPC/private/projects/cmm/roi1_images.npy'
roi1_masks_npy_file = '/data/BIDS-HPC/private/projects/cmm/roi1_masks.npy'
roi2_images_npy_file = '/data/BIDS-HPC/private/projects/cmm/roi2_images.npy'
roi2_masks_npy_file = '/data/BIDS-HPC/private/projects/cmm/roi2_masks.npy'
composite_output_dir = 'composite_aug'
individual_output_dir = 'individual_aug'

# Import general Python modules
import numpy as np
import sys, os, time

# Import our Python modules
sys.path.append(fnlcr_bids_hpc_path+'/packages')
import image_segmentation.utils as imseg_utils # import the utils module
from image_segmentation.image_augmentation import augment_images # import the augment_images function in the image_augmentation module

# Load the data
roi1_images0 = np.load(roi1_images_npy_file)
roi1_masks0 = np.load(roi1_masks_npy_file)
roi2_images0 = np.load(roi2_images_npy_file)
roi2_masks0 = np.load(roi2_masks_npy_file)

# Pad the images for input into the neural networks of nlayers layers
roi1_images = imseg_utils.pad_images(roi1_images0, nlayers)
roi1_masks = imseg_utils.pad_images(roi1_masks0, nlayers)
roi2_images = imseg_utils.pad_images(roi2_images0, nlayers)
roi2_masks = imseg_utils.pad_images(roi2_masks0, nlayers)

# Repeat the images up until the value of neach so that our final images/masks will be a stack of neach*2
nroi1 = roi1_images.shape[0]
nroi2 = roi2_images.shape[0]
reps_roi1 = int(np.ceil(neach/nroi1))
reps_roi2 = int(np.ceil(neach/nroi2))
roi1_images = np.tile(roi1_images, (reps_roi1,1,1))[:neach,:,:]
roi1_masks = np.tile(roi1_masks, (reps_roi1,1,1))[:neach,:,:]
roi2_images = np.tile(roi2_images, (reps_roi2,1,1))[:neach,:,:]
roi2_masks = np.tile(roi2_masks, (reps_roi2,1,1))[:neach,:,:]

# Create the stacks of sizes neach*2
images = np.concatenate((roi1_images, roi2_images))
masks = np.concatenate((roi1_masks, roi2_masks))

# Do the composite augmentations, returning the augmented images and masks as numpy arrays
# Since output_dir is set, we also output .tif files for examination
# num_aug=1 because we manually did the repetitions in our case in order to combine two regions and settle on neach*2 total stack size
# aug_params and composite_sequence will use default values since they are set to None. To customize these, take as examples the settings in fnlcr-bids-hpc/image_segmentation/image_augmentation.py
os.mkdir(composite_output_dir)
time_start = time.time()
#images_aug, masks_aug = augment_images(images, masks=masks, num_aug=1, output_dir=composite_output_dir, imgaug_repo=imgaug_path, aug_params=None, composite_sequence=None)
images_aug, masks_aug = augment_images(images, masks=masks, num_aug=1, output_dir=composite_output_dir)
time_end = time.time()
print('Composite augmentation took {:4.1f} min'.format((time_end-time_start)/60))

# Do the individual augmentations, returning the list of augmented images as numpy arrays
# Since output_dir is set, we also output .tif files for examination
# num_aug=1 because we manually did the repetitions in our case in order to combine two regions and settle on neach*2 total stack size
# aug_params and individual_seqs_and_outnames will use default values since they are set to None. To customize these, take as examples the settings in fnlcr-bids-hpc/image_segmentation/image_augmentation.py
os.mkdir(individual_output_dir)
time_start = time.time()
#images_aug_indiv_list = augment_images(images, num_aug=1, do_composite=False, output_dir=individual_output_dir, imgaug_repo=imgaug_path, aug_params=None, individual_seqs_and_outnames=None)
images_aug_indiv_list = augment_images(images, num_aug=1, do_composite=False, output_dir=individual_output_dir)
time_end = time.time()
print('Individual augmentations took {:4.1f} min total'.format((time_end-time_start)/60))

# Output information about the input and output numpy arrays
imseg_utils.arr_info(roi1_images0)
imseg_utils.arr_info(roi1_masks0)
imseg_utils.arr_info(roi2_images0)
imseg_utils.arr_info(roi2_masks0)
imseg_utils.arr_info(images_aug)
imseg_utils.arr_info(masks_aug)
print(images_aug_indiv_list)