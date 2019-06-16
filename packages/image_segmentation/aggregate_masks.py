# This module contains functions that take multiple (3) inferences of 3D data and combines them to a single set of 3D data, which must be the end result.

def copy_planes(masks, seed):

    # Import relevant modules
    from skimage.measure import label
    import numpy as np

    # Define the return variable
    copies = np.zeros(masks.shape,dtype='bool')

    # For each frame indicated by the first dimension of masks/seed...
    for iframe in range(masks.shape[0]):

        # Determine the connected regions within the masks
        mask_labels, nlabels = label(masks[iframe,:,:].astype('uint8'), return_num=True)

        # For each determined non-background patch...
        copies_frame = np.zeros(mask_labels.shape,dtype='bool') # this is the holder of the intermediate plane copies
        for ilabel in range(nlabels):

            # Define the current patch as that having the current label (ilabel+1)
            patch = mask_labels == (ilabel+1)

            # If the patch overlaps with a seed pixel, assume the current patch is a valid patch (i.e., to be copied over)
            if np.sum(patch & seed[iframe,:,:]) != 0:
                copies_frame[patch] = True
        
        # Save all the valid patches for each frame
        copies[iframe,:,:] = copies_frame
    
    # Return the valid patches
    return(copies)

def get_plane_copies(masks, seed_tuple):
    # inferred_masks[imodel,iinfdir,:,:,:]
    # masks = inferred_masks[imodel,:,:,:,:].astype('bool') --> [3,Z,X,Y] (bool)
    # sums = np.sum(masks, axis=0).astype('uint8') --> [Z,X,Y] (uint8)
    # seed_tuple ~ np.asarray(sums>=2).nonzero() --> (Z,X,Y) (tuple of arrays) (int64) --> goes into something of dimensions [Z,X,Y] like myarr[seed_tuple]

    # Import relevant modules
    import numpy as np

    # Constants
    transpose_indices = ((0,1,2),(1,2,0),(2,0,1)) # corresponds to z, x, y as for other iview-dependent variables
    reverse_transpose_indices = ((0,1,2),(2,0,1),(1,2,0)) # corresponds to z, x, y --> iview-dependent in this case
    par_ind = (2,0,1) # we get this from the index order of z, x, y; "par" stands for "parallel"

    # Define the seed of the same shape as the masks (instead of a tuple of arrays)
    shp = masks.shape
    seed = np.zeros((shp[1],shp[2],shp[3]),dtype='bool')
    seed[seed_tuple] = True # [Z,X,Y] (bool)

    # Initialize the plane-copy-holding variable of interest
    tmp = list(shp)
    tmp.insert(1,3)
    plane_copies = np.zeros(tuple(tmp),dtype='bool') # [3,3,Z,X,Y] --> the first "3" is the view index iview; the second "3" corresponds to the parallel, orth1, and orth2 inferences, respectively

    # For each view in z, x, y order...
    # iview linearly (0,1,2) refers to the (z,x,y) directions, respectively
    for iview in range(3):

        # Get the transpose indices corresponding to the four-dimensional masks array
        tr_ind = list(np.array(transpose_indices[iview])+1)
        tr_ind.insert(0,0)

        # Transpose the arrays of interest so that, for iview = 0,1,2, they are in [Z,X,Y], [X,Y,Z], [Y,Z,X] order
        masks_tr = masks.transpose(tuple(tr_ind))
        seed_tr = seed.transpose(transpose_indices[iview])

        # Get the orthogonal indices since we know which index is the parallel index (terminology note: parallel/orthogonal describes how the inference direction relates to the current view)
        orth_ind = np.setdiff1d((0,1,2),par_ind[iview])

        # Determine the masks corresponding to inference directions that are parallel and orthogonal to the current view
        masks_par = masks_tr[par_ind[iview],:,:,:]
        masks_orth1 = masks_tr[orth_ind[0],:,:,:]
        masks_orth2 = masks_tr[orth_ind[1],:,:,:]

        # Calculate the plane copies
        plane_copies[iview,0,:,:,:] = copy_planes(masks_par, seed_tr).transpose(reverse_transpose_indices[iview])
        plane_copies[iview,1,:,:,:] = copy_planes(masks_orth1, seed_tr).transpose(reverse_transpose_indices[iview])
        plane_copies[iview,2,:,:,:] = copy_planes(masks_orth2, seed_tr).transpose(reverse_transpose_indices[iview])

    # Return the holder of all the plane copies
    return(plane_copies)

def generate_aggregate_masks(inferred_masks_list):
    # This is tested in its own function of the testing module

    # Import relevant modules
    from . import utils
    import numpy as np

    # Constant
    nnew_masks_per_seed = 10 # matches the number of hardcoded offset+ indices below
    copy_types = ['parallel', 'orth-union', 'orth-intersect'] # for the plane copy method

    # Variable to set in this function
    new_masks_list = []

    # For each set of inferred masks...
    for inferred_masks in inferred_masks_list:

        # Get the inferred masks shape, set nmodels correspondingly, and define the longer-term new variable we want to calculate, new_masks
        shp = inferred_masks.shape
        nmodels = shp[0]
        new_masks = np.zeros((nmodels,2*nnew_masks_per_seed,shp[2],shp[3],shp[4]),dtype='bool') # the 2 here corresponds to the number of items in the seeds list below

        # For each model...
        for imodel in range(nmodels):

            # Get the current inferred masks and cast them as Boolean
            masks = inferred_masks[imodel,:,:,:,:].astype('bool')

            # Calculate the corresponding sums, which can be thought of as a generalized intersection
            # sums = 0: none of the three inferences detects a mito; definitely not a mito
            # sums = 1: only one of the inferences detects a mito: probably not a mito
            # sums = 2: only one of the inferences does NOT detect a mito: probably a mito
            # sums = 3: all of the inferences detect a mito: definitely a mito; equals intersection (intersection = masks[0,:,:,:] & masks[1,:,:,:] & masks[2,:,:,:]; ((sums==3) == intersection).all()) is True
            # sums > 0: equals union (union = masks[0,:,:,:] | masks[1,:,:,:] | masks[2,:,:,:]; ((sums>0) == union).all()) is True
            sums = np.sum(masks, axis=0).astype('uint8')
            
            # Define the two types of seeds: (1) that in which the pixel probably corresponds to a mito, and (2) that in which the pixel definitely corresponds to a mito
            seeds = [np.where(sums>=2), np.where(sums==3)]

            # For each seed type...
            new_names = []
            for iseed, seed in enumerate(seeds):

                # Calculate the index offset
                offset = iseed * nnew_masks_per_seed
                seed_str = 'seed' + str(iseed+1)

                # Assign to the longer-term set of new masks the seed-only intermediate set of new masks
                new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                new_masks_one[seed] = True # seed also goes into sums
                new_masks[imodel,offset+0,:,:,:] = new_masks_one
                new_names.append(seed_str+'-'+'seed_only')

                # Run the plane copy method and for each view combine the plane copies three different ways
                plane_copies = get_plane_copies(masks, seed) # [3,3,Z,X,Y] (bool)
                new_masks_three_list = [plane_copies[:,0,:,:,:]]
                new_masks_three_list.append(plane_copies[:,1,:,:,:] | plane_copies[:,2,:,:,:])
                new_masks_three_list.append(plane_copies[:,1,:,:,:] & plane_copies[:,2,:,:,:])

                # For each of three sets of plane copy results ("three"), combine them in three different ways to result in a single set of masks ("one")
                for offset_index, new_masks_three in enumerate(new_masks_three_list):

                    # Calculate the index offset
                    offset2 = offset_index * 3

                    # Start combining the plane copy results by summing as before
                    masks = new_masks_three.astype('bool')
                    sums = np.sum(masks, axis=0).astype('uint8')

                    # Assign to the longer-term set of new masks the current plane-copy method + union set of new masks
                    new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                    new_masks_one[np.where(sums>=1)] = True
                    new_masks[imodel,offset+offset2+1,:,:,:] = new_masks_one
                    new_names.append(seed_str+'-'+copy_types[offset_index]+'-'+'ge_1')

                    # Assign to the longer-term set of new masks the current plane-copy method + >=2 set of new masks
                    new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                    new_masks_one[np.where(sums>=2)] = True
                    new_masks[imodel,offset+offset2+2,:,:,:] = new_masks_one
                    new_names.append(seed_str+'-'+copy_types[offset_index]+'-'+'ge_2')

                    # Assign to the longer-term set of new masks the current plane-copy method + intersection set of new masks
                    new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                    new_masks_one[np.where(sums==3)] = True
                    new_masks[imodel,offset+offset2+3,:,:,:] = new_masks_one
                    new_names.append(seed_str+'-'+copy_types[offset_index]+'-'+'eq_3')

        # Add this longer-term set of new masks to the set of new masks (a list) that we'll actually return
        new_masks_list.append(new_masks)

    # Return the new masks list and the names of the new aggregate masks
    return(new_masks_list, new_names)

def make_movies_aggregate_masks(roi_list, images_list, new_masks_list, models, nframes=40, known_masks_list=None, metrics_2d_list=None, metrics_3d_list=None, framerate=2, delete_frames=True, ffmpeg_preload_string='module load FFmpeg; ', new_names=None):

    # Import relevant modules
    import numpy as np
    import matplotlib.pyplot as plt
    from . import utils
    import os
    import glob

    # Constants
    transpose_indices = ((0,1,2),(1,2,0),(2,0,1)) # corresponds to z, x, y as for other iview-dependent variables
    legend_arr = ['true positive rate / sensitivity / recall','true negative rate / specificity / selectivity','positive predictive value / precision','balanced accuracy','f1 score']
    do_transpose_for_view = [False,True,True] # z, x, y, as normal for views
    labels_views = ['Z','X','Y']

    # Create the movies directory if it doesn't already exist
    if not os.path.exists('movies'):
        os.mkdir('movies')

    # Determine whether we have a situation in which the true masks (and therefore metrics) are known
    if known_masks_list is None:
        truth_known = False
    else:
        truth_known = True

    # For every set of images, new masks, and, if applicable, known masks and metrics...
    ilist = 0
    for images, new_masks, roi in zip(images_list, new_masks_list, roi_list):
        if truth_known:
            known_masks = known_masks_list[ilist]
            metrics_2d = metrics_2d_list[ilist]
            metrics_3d = metrics_3d_list[ilist]
            nmetrics = metrics_3d.shape[2]

        # Set some variables needed later
        shp = new_masks.shape
        nmodels = shp[0]
        nnew_masks = shp[1]
        unpadded_shape = shp[2:]
        nviews = 3

        # For every model...
        for imodel in range(nmodels):
            print('On model '+str(imodel+1)+' of '+str(nmodels)+' ('+models[imodel]+')')

            # For every view...
            for iview in range(nviews):
                print('  On view '+str(iview+1)+' of '+str(nviews))

                # Clear the figure if it exists
                plt.clf()

                # Get the current data
                curr_stack_size = unpadded_shape[iview]
                curr_images = images.transpose(transpose_indices[iview])
                if truth_known:
                    curr_known_masks = known_masks.transpose(transpose_indices[iview])

                # For each truly 3D mask...
                for inew_mask in range(nnew_masks):
                    print('    On new mask '+str(inew_mask+1)+' of '+str(nnew_masks))
                
                    # Obtain the current data
                    curr_new_masks = np.squeeze(new_masks[imodel,inew_mask,:,:,:]).transpose(transpose_indices[iview])
                    if truth_known:
                        curr_metrics_2d = np.squeeze(metrics_2d[imodel,inew_mask,:,iview,:curr_stack_size])
                        curr_metrics_3d = np.squeeze(metrics_3d[imodel,inew_mask,:])

                    # Determine the figure size (and correspondingly, the subplots size)
                    fig_width = 6 # inches
                    nsp_cols = 1 # sp = subplot
                    if truth_known:
                        fig_height = 9
                        nsp_rows = 2
                    else:
                        fig_height = 5
                        nsp_rows = 1

                    # Set the figure size
                    plt.figure(figsize=(fig_width,fig_height)) # interestingly you must initialize figsize here in order to make later calls to myfig.set_figwidth(X) work

                    # Set the subplots size and get the axes handles
                    ax_images = plt.subplot(nsp_rows,nsp_cols,1)
                    if truth_known:
                        ax_metrics = plt.subplot(nsp_rows,nsp_cols,1+1)
                    
                    # Frame-independent plotting
                    ax_images.set_title('view='+labels_views[iview])
                    new_mask_name = models[imodel].split('-',1)[0]+'-'+new_names[inew_mask]
                    if truth_known:
                        ax_metrics.plot(curr_metrics_2d.transpose())
                        ax_metrics.set_xlim(0,curr_stack_size-1)
                        ax_metrics.set_ylim(0,1)
                        ax_metrics.set_xlabel('3D stats: tpr='+'{:04.2f}'.format(curr_metrics_3d[0])+', tnr='+'{:04.2f}'.format(curr_metrics_3d[1])+', ppv='+'{:04.2f}'.format(curr_metrics_3d[2])+', bacc='+'{:04.2f}'.format(curr_metrics_3d[3])+', f1='+'{:04.2f}'.format(curr_metrics_3d[4]))
                        ax_metrics.set_ylabel(new_mask_name)
                        ax_metrics.legend(legend_arr,loc='lower left')
                  
                    # Determine if for the current view we should rotate the 2D plot by 90 degrees
                    if do_transpose_for_view[iview]:
                        rotate_2d = (1,0,2)
                    else:
                        rotate_2d = (0,1,2)

                    # Now plot the frame-dependent data and metrics...for every frame...
                    for frame in np.linspace(0,curr_stack_size-1,num=nframes).astype('uint16'):
                        print('    On frame '+str(frame+1)+' in '+str(curr_stack_size))

                        # Set variables that are the same for each inference direction: curr_images_frame, (curr_known_masks_frame)
                        curr_images_frame = np.transpose(np.squeeze(utils.arr2rgba(curr_images[frame,:,:],A=1*255,shade_color=[1,1,1],makeBGTransp=False)),rotate_2d)
                        if truth_known:
                            curr_known_masks_frame = np.transpose(np.squeeze(utils.arr2rgba(curr_known_masks[frame,:,:],A=0.2*255,shade_color=[0,0,1],makeBGTransp=True)),rotate_2d)

                        # Set variables that are different for each inference direction (ax_images, (ax_metrics), curr_new_masks_frame, (curr_metrics_2d_frame)) and do the plotting
                        temporary_plots = []
                        curr_new_masks_frame = np.transpose(np.squeeze(utils.arr2rgba(curr_new_masks[frame,:,:],A=0.2*255,shade_color=[1,0,0],makeBGTransp=True)),rotate_2d)
                        temporary_plots.append(ax_images.imshow(curr_images_frame))
                        temporary_plots.append(ax_images.imshow(curr_new_masks_frame))
                        if truth_known:
                            curr_metrics_2d_frame = np.squeeze(curr_metrics_2d[:,frame])
                            ax_metrics.set_title('tpr='+'{:04.2f}'.format(curr_metrics_2d_frame[0])+' tnr='+'{:04.2f}'.format(curr_metrics_2d_frame[1])+' ppv='+'{:04.2f}'.format(curr_metrics_2d_frame[2])+' bacc='+'{:04.2f}'.format(curr_metrics_2d_frame[3])+' f1='+'{:04.2f}'.format(curr_metrics_2d_frame[4]))
                            temporary_plots.append(ax_images.imshow(curr_known_masks_frame))
                            temporary_plots.append(ax_metrics.scatter(np.ones((nmetrics,1))*frame,curr_metrics_2d_frame,c=['C0','C1','C2','C3','C4']))

                        # Save the figure to disk
                        plt.savefig('movies/'+roi+'__model_'+new_mask_name+'__view_'+labels_views[iview]+'__frame_'+'{:04d}'.format(frame)+'.png',dpi='figure')

                        # Delete temporary objects from the plot
                        for temporary_plot in temporary_plots:
                            temporary_plot.set_visible(False)

                    # Determine the string that is a glob of all the frames
                    frame_glob = 'movies/'+roi+'__model_'+new_mask_name+'__view_'+labels_views[iview]+'__frame_'+'*'+'.png'

                    # Create the movies
                    os.system(ffmpeg_preload_string + 'ffmpeg -r '+str(framerate)+' -pattern_type glob -i "'+frame_glob+'" -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -movflags faststart ' + 'movies/'+roi+'__model_'+new_mask_name+'__view_'+labels_views[iview]+'.mp4')

                    # Unless otherwise specified, delete the frames
                    if delete_frames:
                        for frame in glob.glob(frame_glob):
                            os.remove(frame)
                    
        # Go to the next set of images etc. for the current ROI
        ilist += 1