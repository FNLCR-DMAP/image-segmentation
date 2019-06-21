
from skimage import io
import os
from mask_feature import generate_mask_features
import matplotlib.pyplot as plt

gt_dir="/data/HiTIF/data/dl_segmentation_input/HiTIF_colorectal/ground_truth/exp2"
file_name="AssayPlate_PerkinElmer_CellCarrier-384 Ultra_B07_T0001F002L01A01Z01C01.tif"
gt_file=os.path.join(gt_dir, file_name)
masks = io.imread(gt_file)

feat_dict = {"bitmask":True, "distance_transform":None, "erosion":True,"edge":True, "blured_contour":None}

features = generate_mask_features(masks, **feat_dict)
fig, axes = plt.subplots(1,6, figsize=(40,40))
subset=200
axes[0].imshow(features["bitmask"][0:200, 0:200], cmap='jet')
axes[1].imshow(features["distance_transform"][0:200, 0:200], cmap='jet')
axes[2].imshow(features["erosion"][0:200, 0:200], cmap='jet')
axes[3].imshow(features["edge"][0:200, 0:200], cmap='jet')
axes[4].imshow(features["blured_contour"][0:200, 0:200], cmap='jet')
plt.show()
_ = raw_input("Press [enter] to continue.")     
print features
