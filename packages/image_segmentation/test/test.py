# In Bash (only on Biowulf): module load python/3.6
import sys
#sys.path.append('/home/weismanal/checkouts/fnlcr-bids-hpc/packages') # only on Biowulf
#sys.path.append('/Users/weismanal/checkouts/fnlcr-bids-hpc/packages') # only on laptop
from image_segmentation.test import testing

# Test the helper functions in the testing module
#print(testing.define_data())
#print(testing.load_data())

# Test the image_augmentation module
#testing.testing__image_augmentation__augment_images()

# Test the post_inference module
#testing.testing__post_inference__load_inferred_masks()
#testing.testing__post_inference__calculate_metrics()
#testing.testing__post_inference__output_metrics()
#testing.testing__post_inference__make_movies_individual_masks()

# Test the aggregate_masks module
#testing.testing__aggregate_masks__copy_planes()
#testing.testing__aggregate_masks__get_plane_copies()
#testing.testing__aggregate_masks__generate_aggregate_masks()
#testing.testing__aggregate_masks__make_movies_aggregate_masks()