def define_data():
    # This is the only place where values should be hardcoded

    #roi_list = ['roi3','roi3']
    roi_list = ['roi3']
    # The following list of models is created by running in Bash: tmp=$(find . -type d -iregex "\./[0-9][0-9]-hpset.*" | sort | awk -v FS="./" -v ORS="','" '{print $2}'); models="['${tmp:0:${#tmp}-2}]"; echo $models
    #models = ['01-hpset_10','02-hpset_11','03-hpset_16','04-hpset_17','05-hpset_21a','06-hpset_21b','07-hpset_21c','08-hpset_21d','09-hpset_22','10-hpset_23','11-hpset_28','12-hpset_30','13-hpset_32','14-hpset_33','15-hpset_34','16-hpset_last_good_unet','17-hpset_resnet']
    #models = ['08-hpset_21d','09-hpset_22']
    models = ['08-hpset_21d']
    inference_directions = ['x','y','z']
    #data_dir = '/home/weismanal/notebook/2019-02-13'
    data_dir = '/Users/weismanal/notebook/2019-04-03/testing_three_modules/data'

    return(roi_list, data_dir, models, inference_directions)

def load_data():
    # Loads data specifically for testing

    # Import relevant modules
    from ..post_inference import load_inferred_masks
    import numpy as np
    from .. import utils

    # Get data definitions
    roi_list, data_dir, models, inference_directions = define_data()

    # Load all the data
    images_list = []
    known_masks_list = []
    inferred_masks_list = []
    for roi in roi_list:
        images_list.append(utils.normalize_images(np.load(data_dir+'/'+roi+'_input_img.npy'), 1, do_output=False)) # normalize to uint8
        known_masks_list.append(utils.normalize_images(np.load(data_dir+'/'+'known_masks_'+roi+'.npy'), 1, do_output=False)) # normalize to uint8
        inferred_masks_list.append(load_inferred_masks(roi, images_list[-1].shape, models, inference_directions, data_dir, do_output=False)) # these are ultimately uint8

    # Return all the data
    return(images_list, known_masks_list, inferred_masks_list, roi_list, models)

def testing__image_augmentation__augment_images():
    # Tests image_augmentation.augment_images()

    # Instructions on installing and running the module are located at https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages/image_segmentation/

    # Import relevant modules
    from ..image_augmentation import augment_images
    from skimage import io
    import numpy as np
    from .. import utils
    import os

    # Constant
    imgaug_path = '/data/BIDS-HPC/public/software/checkouts/imgaug'
    #imgaug_path = '/Users/weismanal/checkouts/imgaug'
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # Input image definitions
    lady_single_rgb = io.imread(curr_dir + '/../sample_data/lady_rgb.jpg') # (H,W,3)
    lady_stack_rgb = np.load(curr_dir + '/../sample_data/lady_images_rgb_original_15.npy') # (N,H,W,3)
    lady_single_gray = lady_single_rgb[:,:,0] # (H,W)
    lady_stack_gray = lady_stack_rgb[:,:,:,0] # (N,H,W)
    lady_stack_masks = np.load(curr_dir + '/../sample_data/lady_masks_original_15.npy') # (N,H,W)

    # Augment images only
    mydir = 'augmented_images_only'
    utils.create_dir(mydir)
    # image_aug = augment_images(lady_single_gray, num_aug=50)
    # image_aug = augment_images(lady_single_rgb)
    # image_aug = augment_images(lady_single_rgb, num_aug=25, output_dir=mydir)
    # image_aug = augment_images(lady_single_rgb, num_aug=50)
    # image_aug = augment_images(lady_single_rgb, output_dir=mydir)
    # image_aug = augment_images(lady_stack_gray)
    # image_aug = augment_images(lady_stack_rgb)
    image_aug = augment_images(lady_stack_rgb, output_dir=mydir)
    print(image_aug.shape)

    # Augment both images and masks
    mydir = 'augmented_images_and_masks'
    utils.create_dir(mydir)
    # image_aug, mask_aug = augment_images(lady_stack_gray, masks=lady_stack_masks)
    # image_aug, mask_aug = augment_images(lady_stack_gray, masks=lady_stack_masks, output_dir=mydir)
    # image_aug, mask_aug = augment_images(lady_stack_gray, num_aug=3, masks=lady_stack_masks)
    # image_aug, mask_aug = augment_images(lady_stack_rgb, masks=lady_stack_masks)
    image_aug, mask_aug = augment_images(lady_stack_rgb, masks=lady_stack_masks, output_dir=mydir)
    # image_aug, mask_aug = augment_images(lady_stack_rgb, num_aug=3, masks=lady_stack_masks)
    print(image_aug.shape, mask_aug.shape)

    # View individual image augmentations
    mydir = 'individual_augmentations'
    utils.create_dir(mydir)
    # image_aug_list = augment_images(lady_single_rgb, do_composite=False)
    # image_aug_list = augment_images(lady_single_rgb, do_composite=False, output_dir=mydir)
    # image_aug_list = augment_images(lady_single_rgb, num_aug=5, do_composite=False, output_dir=mydir)
    # image_aug_list = augment_images(lady_stack_gray, do_composite=False)
    # image_aug_list = augment_images(lady_stack_rgb, do_composite=False)
    # image_aug_list = augment_images(lady_stack_rgb, do_composite=False, output_dir=mydir)
    image_aug_list = augment_images(lady_stack_rgb, num_aug=2, do_composite=False, output_dir=mydir)
    print(image_aug_list[0].shape)

def testing__post_inference__load_inferred_masks():
    # Tests post_inference.load_inferred_masks()

    # Load necessary data
    roi_list, data_dir, models, inference_directions = define_data()

    # Test function of interest
    from ..post_inference import load_inferred_masks
    import numpy as np
    inferred_masks_list = []
    for roi in roi_list:
        inferred_masks_list.append(load_inferred_masks(roi, np.load(data_dir+'/'+roi+'_input_img.npy').shape, models, inference_directions, data_dir, do_output=False)) # these are ultimately uint8

    # Check output by eye
    from .. import utils
    utils.arr_info(inferred_masks_list[0])

def testing__post_inference__calculate_metrics():
    # Tests post_inference.calculate_metrics()

    # Load necessary data
    __, known_masks_list, inferred_masks_list, __, __ = load_data()

    # Test the function of interest
    from ..post_inference import calculate_metrics
    metrics_2d_list, __ = calculate_metrics(known_masks_list, inferred_masks_list)

    # Just make sure this output looks reasonable; if so, the rest is probably good too
    import numpy as np
    print(np.squeeze(metrics_2d_list[0][0,2,0,:,:]))

def testing__post_inference__output_metrics():
    # Tests post_inference.output_metrics()

    # Load necessary data
    __, known_masks_list, inferred_masks_list, roi_list, models = load_data()
    from ..post_inference import calculate_metrics
    __, metrics_3d_list = calculate_metrics(known_masks_list, inferred_masks_list)

    # Test function of interest; it will output metrics_3d.html and metrics_3d.txt
    from ..post_inference import output_metrics
    output_metrics(metrics_3d_list, roi_list, models)

def testing__post_inference__make_movies_individual_masks():
    # Tests post_inferece.make_movies_individual_masks()

    # Load necessary data
    images_list, known_masks_list, inferred_masks_list, roi_list, models = load_data()
    from ..post_inference import calculate_metrics
    metrics_2d_list, metrics_3d_list = calculate_metrics(known_masks_list, inferred_masks_list)
    
    # Test function of interest; it will output multiple .mp4 movies
    from ..post_inference import make_movies_individual_masks
    make_movies_individual_masks(roi_list, images_list, inferred_masks_list, models, nframes=40, known_masks_list=known_masks_list, metrics_2d_list=metrics_2d_list, metrics_3d_list=metrics_3d_list)

def testing__aggregate_masks__copy_planes():
    # Tests aggregate_masks.copy_planes()

    # Load necessary data
    __, __, inferred_masks_list, __, __ = load_data()

    # Test function of interest
    from ..aggregate_masks import copy_planes
    plane_copy = copy_planes(inferred_masks_list[0][0,0,:,:,:], inferred_masks_list[0][0,0,:,:,:] & inferred_masks_list[0][0,1,:,:,:] & inferred_masks_list[0][0,2,:,:,:]) # copy_planes(masks, seed)

    # Check by eye
    from .. import utils
    utils.arr_info(plane_copy)

def testing__aggregate_masks__get_plane_copies():
    # Tests aggregate_masks.get_plane_copies()

    # Load necessary data
    __, __, inferred_masks_list, __, __ = load_data()

    # Test function of interest
    masks = inferred_masks_list[0][0,:,:,:,:].astype('bool')
    from ..aggregate_masks import get_plane_copies
    import numpy as np
    plane_copies = get_plane_copies(masks, np.where(np.sum(masks, axis=0).astype('uint8')>=2)) # [3,3,Z,X,Y] (bool); get_plane_copies(masks, seed_tuple)

    # Check by eye
    from .. import utils
    utils.arr_info(plane_copies)

def testing__aggregate_masks__generate_aggregate_masks():
    # Tests aggregate_masks.generate_aggregate_masks()

    # Load necessary data
    __, __, inferred_masks_list, __, __ = load_data()

    # Test function of interest
    from ..aggregate_masks import generate_aggregate_masks
    aggregate_masks_list, new_names = generate_aggregate_masks(inferred_masks_list)

    # Check by eye
    from .. import utils
    utils.arr_info(aggregate_masks_list[0])
    print(new_names)

def testing__aggregate_masks__make_movies_aggregate_masks():
    # Tests aggregate_masks.make_movies_aggregate_masks()

    # Load necessary data
    images_list, known_masks_list, inferred_masks_list, roi_list, models = load_data()
    from ..aggregate_masks import generate_aggregate_masks
    aggregate_masks_list, new_names = generate_aggregate_masks(inferred_masks_list)
    from ..post_inference import calculate_metrics
    metrics_2d_list, metrics_3d_list = calculate_metrics(known_masks_list, aggregate_masks_list, nviews=3)
    
    # Test the function of interest; this should generate multiple .mp4 movies in a new movies directory
    from ..aggregate_masks import make_movies_aggregate_masks
    make_movies_aggregate_masks(roi_list, images_list, aggregate_masks_list, models, nframes=40, known_masks_list=known_masks_list, metrics_2d_list=metrics_2d_list, metrics_3d_list=metrics_3d_list, new_names=new_names)

    # Output the 3D metrics .html and .txt files to check
    from ..post_inference import output_metrics
    output_metrics(metrics_3d_list, roi_list, models, new_names=new_names)

    # Also output some metrics to check by eye
    import numpy as np
    print(np.squeeze(metrics_2d_list[0][0,1,0,:,:]))