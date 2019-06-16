# This module should contain relatively simple functions that can be used for multiple image segmentation-related purposes
# "Compound" functions that call the functions in this script (among other things) should be part of other modules in the image_segmentation package

def arr_info(arr,mystr=''):
    # Print information about a numpy array
    import numpy as np
    print('-------------- '+mystr+'Printing array info of numpy array --------------')
    print('Shape: ',arr.shape)
    print('dtype: ',arr.dtype)
    print('Min: ',np.min(arr))
    print('Max: ',np.max(arr))
    hist,edges = np.histogram(arr)
    print('Hist: ',hist)
    print('Edges: ',edges)

def arr2rgba(arr,A=255,shade_color=[1,1,1],makeBGTransp=False):
    # By default, make the image not transparent at all, make the transparency color white, and don't make the 0-pixels (of the first channel) transparent
    # Fourth channel = 255 means no transparency (fully opaque); fourth channel = 0 means fully transparent
    # Calling it "arr" to be clear that the input can be either an image or a mask
    import numpy as np

    # Ensure the shade color is a float
    shade_color = np.array(shade_color,dtype='float32')
    
    # Preprocess the images so that they're uint8 and (N,H,W,C)
    #rgba = np.expand_dims(arr,3) # (N,H,W) --> (N,H,W,1)
    arr = normalize_images(arr,1,do_output=False) # ensure the images are uint8; inputs can therefore be [0,1], [0,2^8-1], or [0,2^16-1]
    arr, __ = stack_and_color_images(arr) # (N,H,W) --> (N,H,W,1)... now arr is always going to be (N,H,W,C) no matter what the input format, which used to have to be (N,H,W)

    # If gray, set the new array (rgba) to an RGB copy of it; otherwise (RGB), set the new array to a copy of it
    # rgba = np.tile(rgba,(1,1,1,4)) # --> (N,H,W,4)
    if arr.shape[3] == 1:
        rgba = np.tile(arr,(1,1,1,3)) # --> (N,H,W,3)
    else:
        rgba = np.copy(arr) # --> (N,H,W,3)
    
    # Tack on the transparency array to the new array    
    shp = list(rgba.shape[0:3])
    shp.append(1)
    transp = np.ones(tuple(shp),dtype='uint8') * int(A)
    rgba = np.concatenate((rgba,transp),axis=3)

    # Blacken out the three channels according to shade_color (i.e., "shade" the array)
    # shade_color=[1,1,1] leaves it unchanged
    # shade_color=[0,0,0] makes it completely black
    rgba[:,:,:,0] = (rgba[:,:,:,0]*shade_color[0]).astype('uint8')
    rgba[:,:,:,1] = (rgba[:,:,:,1]*shade_color[1]).astype('uint8')
    rgba[:,:,:,2] = (rgba[:,:,:,2]*shade_color[2]).astype('uint8')
    #rgba[:,:,:,3] = A # A=255 means no transparency

    # If set, if the first channel is zero, make the corresponding values completely transparent
    if makeBGTransp:
        bg0,bg1,bg2 = np.where(arr[:,:,:,0]==0)
        rgba[bg0,bg1,bg2,3] = 0

    # Return this new array
    return(rgba)

def calculate_metrics(msk0,msk1,twoD_stack_dim=-1):
    # Calculate how well the image segmentation task performed by comparing the inferred masks (msk1) to the known masks (msk0)

    # Import relevant modules
    import numpy as np

    # Arrays required for calculations of metrics
    target = msk0.astype('bool')
    guess = msk1.astype('bool')
    overlap_fg = target & guess
    overlap_bg = (~target) & (~guess)

    # Process the arrays for the true number of foreground and background pixels and those the model gets correct
    nfg = count(target,twoD_stack_dim=twoD_stack_dim)
    nbg = count(~target,twoD_stack_dim=twoD_stack_dim)
    noverlap_fg = count(overlap_fg,twoD_stack_dim=twoD_stack_dim)
    noverlap_bg = count(overlap_bg,twoD_stack_dim=twoD_stack_dim)

    # Convert to true/false positives/negatives
    npos = nfg # purple + blue
    nneg = nbg # gray + red
    ntruepos = noverlap_fg # --> should be number of purples
    ntrueneg = noverlap_bg # --> should be number of grays
    nfalsepos = nneg - ntrueneg # --> should be number of reds (reason this makes sense: a false positive is really a negative that's not a true negative)
    nfalseneg = npos - ntruepos # --> should be number of blues (reason this makes sense: a false negative is really a positive that's not a true positive)

    # Metrics that should all be large
    tpr = ntruepos / (ntruepos + nfalseneg) # = ntruepos / npos # i.e., how good you are at detecting mito (sensitivity = recall = true positive rate)
    tnr = ntrueneg / (ntrueneg + nfalsepos) # = ntrueneg / nneg # i.e., how good you are at detecting NOT mito (specificity = selectivity = true negative rate)
    ppv = ntruepos / (ntruepos + nfalsepos) # positive predictive value = precision (i.e., of all the red that you see, how much of it is correct?)
    bacc = (tpr+tnr) / 2 # Overall accuracy (balanced accuracy)
    f1 = 2 / ( (1/tpr) + (1/ppv) )
    
    # Return a numpy array of the metrics
    return(np.array([tpr,tnr,ppv,bacc,f1]))
    
def count(arr,twoD_stack_dim=-1):
    # Sum an array over particular dimensions (over all dimensions by default)
    import numpy as np
    dims = [0,1,2]
    if twoD_stack_dim == -1:
        return(np.sum(arr))
    else:        
        dims.remove(twoD_stack_dim)
        return(np.sum(arr,axis=tuple(dims)))

# Define a function to create paths in the working directory
def create_dir(path):
    import os
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)

def get_colored_str(x):
    # Get HTML string that colors x according to its value so the table is colored
    col = 'black'
    if x >= 95:
        col = 'green'
    elif x >= 85:
        col = 'orange'
    elif x >= 75:
        col = 'red'
    numstr = '{: 4d}'.format(x)
    return('<font style="color:'+col+';">'+numstr+'</font>')

def normalize_images_orig(images,idtype,do_output=True):
    # Automatically detect the pixel range and scale the images to a new range specified by idtype
    import numpy as np
    possible_range_maxs = [1,2**8-1,2**16-1]
    possible_range_dtypes = ['float32','uint8','uint16']
    median = np.median(images)
    if do_output:
        print('')
        arr_info(images,mystr="BEFORE: ")
    found = False
    range_min = 0
    for range_max in possible_range_maxs:
        if (median>=range_min) and (median<=range_max):
            found = True
            break
        range_min = range_max
    if not found:
        print('No range found')
        print(median)
        arr_info(images)
        return(-1)
    else:
        if do_output:
            print('Range found; median is ',median,' and range max is ',range_max)
        images2 = ( images.astype('float32') / range_max * possible_range_maxs[idtype] ).astype(possible_range_dtypes[idtype])
        if do_output:
            if (images2==images).all():
                print('UNCHANGED')
            else:
                arr_info(images2,mystr="AFTER: ")
        return(images2)
        
def normalize_images(images,idtype,do_output=True):
    # Automatically detect the pixel range and scale the images to a new range specified by idtype
    import numpy as np
    possible_range_maxs = [1,2**8-1,2**16-1]
    possible_range_dtypes = ['float32','uint8','uint16']
    mymax = np.max(images)
    if do_output:
        print('')
        arr_info(images,mystr="BEFORE: ")
    found = False
    for range_max in possible_range_maxs:
        if mymax <= range_max:
            found = True
            break
    if not found:
        print('No range found')
        print(mymax)
        arr_info(images)
        return(-1)
    else:
        if do_output:
            print('Range found; max is ',mymax,' and range max is ',range_max)
        images2 = ( images.astype('float32') / range_max * possible_range_maxs[idtype] ).astype(possible_range_dtypes[idtype])
        if do_output:
            if (images2==images).all():
                print('UNCHANGED')
            else:
                arr_info(images2,mystr="AFTER: ")
        return(images2)
        
def overlay_images(rgba1, rgba2):
    # Inputs are necessarily (N,H,W,4) and uint8 since they should be first run through arr2rgba()
    # Returns (N,H,W,3) uint8
    import numpy as np
    rgb1 = rgba1[:,:,:,0:3]
    rgb2 = rgba2[:,:,:,0:3]
    transp = np.tile(np.expand_dims(rgba2[:,:,:,3],3),(1,1,1,3)) / 255
    return( ( (1-transp)*rgb1 + transp*rgb2 ).astype('uint8') )

def pad_images(images, nlayers):
    """
    In Unet, every layer the dimension gets divided by 2
    in the encoder path. Therefore the image size should be divisible by 2^nlayers.
    """
    import math
    import numpy as np
    divisor = 2**nlayers
    nlayers, x, y = images.shape # essentially setting nlayers to z direction so return is z, x, y
    x_pad = int((math.ceil(x / float(divisor)) * divisor) - x)
    y_pad = int((math.ceil(y / float(divisor)) * divisor) - y)
    padded_image = np.pad(images, ((0,0),(0, x_pad), (0, y_pad)), 'constant', constant_values=(0, 0))
    return padded_image

def quick_overlay_output(images, masks, overlay_tif_path):
    from skimage import io
    rgba1 = arr2rgba(images,A=255,shade_color=[1,1,1],makeBGTransp=False)
    rgba2 = arr2rgba(masks,A=round(0.25*255),shade_color=[1,0,0],makeBGTransp=True)
    io.imsave(overlay_tif_path,overlay_images(rgba1, rgba2))

def randomize_labels(labels):
    import random
    import numpy as np
    random.seed(1)
    ind_labels = np.nonzero(labels!=0) # goes into labels; get the indices of labels that aren't background (=ind above)
    labels_nonzero = labels[ind_labels] # get the foreground labels
    labels2 = np.copy(labels) # duplicate the original labels
    y = np.unique(labels_nonzero) # same as above
    nx = y.size + 1 # same as above
    z = y + 10*nx # same as above
    random.shuffle(z) # shuffle the shifted unique labels (same as above)    
    for i in np.arange(0,nx-1): # for indices of the unique labels excluding 0...
        ilabel = y[i]
        iz = z[i]
        ind_labels_nonzero = np.nonzero(labels_nonzero==ilabel) # goes into labels_nonzero and ind_labels[X]; determine where the foreground labels equal the current unique foreground label
        ind0 = ind_labels[0][ind_labels_nonzero] # get the "x" indices of labels that are the current foreground value
        ind1 = ind_labels[1][ind_labels_nonzero] # get the "y" indices of labels that are the current foreground value
        ind2 = ind_labels[2][ind_labels_nonzero] # get the "z" indices of labels that are the current foreground value
        labels2[ind0,ind1,ind2] = iz
        #print(labels2[ind0,ind1,ind2]) # print the shifted and randomized foreground values that are the current unshifted, unrandomized foreground value
    labels2[ind_labels] = labels2[ind_labels]  - 10*nx    
    return(labels2)

def stack_and_color_images(images, masks=None):
    # images can be (H,W), (H,W,3), (N,H,W), or (N,H,W,3)
    # masks can be (H,W) or (N,H,W)
    # At the end of this function, images should be (N,H,W,C) and masks should be (N,H,W)
    import numpy as np

    # If images are not stacked...
    if (images.ndim == 2) or (images.ndim==3 and images.shape[2]==3): # (H,W) or (H,W,C)
        images = np.expand_dims(images,0)
        if masks is not None:
            masks = np.expand_dims(masks,0)

    # If images are not colored...
    if images.ndim == 3: # (N,H,W)
        images = np.expand_dims(images,3)

    return(images, masks)

def transpose_stack(images):
    # Split the images into all three dimensions, making the other two dimensions of the images accessible
    # Input images should be three dimensions (e.g., a stack)
    x_first = images.transpose((1,2,0)) # (x,y,z)
    y_first = images.transpose((2,0,1)) # (y,z,x)
    #z_first = images.transpose((0,1,2)) # (z,x,y)
    z_first = images
    return(x_first,y_first,z_first)