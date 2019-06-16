
import numpy as np
from image_augmentation import augment_images
import utils
from skimage import io
import os
import augment_hitif
from augment_hitif import hitif_aug

augmenter = hitif_aug("configuration-well07.cfg")

file_name="Row0-NNI-1by1.tif"
fov_dir="."
gt_dir="."

image_file=os.path.join(fov_dir, file_name)
gt_file=os.path.join(gt_dir, file_name)


images = io.imread(image_file)[:255, :255]
masks = io.imread(gt_file)[:255, :255]

print("Images")
utils.arr_info(images)


print("MASKS")
utils.arr_info(masks)

print(images.shape)
print(masks.shape)

augment_images(images, masks, 1, do_composite=False, AugSettings= augmenter, output_dir="augmented-samples-well07")
augment_images(images, masks, 1, do_composite=False, AugSettings= augmenter, output_dir="augmented-samples-well07")
#image_aug, mask_aug = augment_images(images, masks, 200, do_composite=True, AugSettings= augmenter, output_dir="augmented-samples-well07")

#
#io.imsave('augmented.tif',image_aug)
#io.imsave('masks.tif',mask_aug)
#rgba1 = utils.arr2rgba(image_aug,A=255,shade_color=[1,1,1],makeBGTransp=False)
#rgba2 = utils.arr2rgba(mask_aug,A=round(0.25*255),shade_color=[1,0,0],makeBGTransp=True)
#io.imsave('output-overlay.tif',utils.overlay_images(rgba1, rgba2))
