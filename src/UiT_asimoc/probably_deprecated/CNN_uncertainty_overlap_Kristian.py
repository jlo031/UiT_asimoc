import numpy as np
import time
from torch.cuda import is_available

def Uncertainty_overlapping_patches(model, image_test, patch_size, device = '', stride_overlap = 1, n_classes = 2):

    # Only valid for binary classifier.    



    # Set model to evaluation mode

    model.eval()

    
    # Extract dimensions of test image

    y_dim, x_dim = image_test.shape[2:]

    
    # Determine number of patches in each direction

    n_patches_x = int(np.floor((x_dim-patch_size)/stride_overlap)) + 1

    n_patches_y = int(np.floor((y_dim-patch_size)/stride_overlap)) + 1

    
    # Generate variables to store results of each patch

    # Initialize to -1, while the outout should be 0 and 1 only for

    # the class_mat, while the probs_mat should be [0.5,1], as this

    # keeps the probability of the most probable class for a binary

    # classifier.

    class_mat = np.zeros( (n_patches_y,  n_patches_x), dtype='int8') - 1

    probs_mat = np.zeros( (n_patches_y,  n_patches_x)) - 1

    # These are really not needed, so they can removed if wanted.

    
    # Generate outputs per-pixel

    overlaps       = np.zeros( (y_dim,  x_dim), dtype='int8')

    class_mat_full = np.zeros( (y_dim,  x_dim), dtype='float32')

    probs_mat_full = np.zeros( (y_dim,  x_dim), dtype='float32')

    # 'overlaps' in the number of overlapping patches for each pixel

    # class_mat_full is the number of times the of the 1-class per pixel

    # probs_mat_full is the mean probability of the 1-class per pixel

    # The '1-class' is the second class, as the first class is the 0-class.


    # Check device and send to device

    if is_available():

        device = "cuda"

    else:

        device = "cpu"

    model.to(device=device)



    # Loop over all patches

    since = time.time()

    for i in range(n_patches_x):

        x_curr = i*stride_overlap

        for j in range(n_patches_y):

            y_curr =j*stride_overlap

            # Extract current patch

            patch = image_test[:,:,y_curr:(y_curr+patch_size) , x_curr:(x_curr+patch_size)]

                
            # Run forward pass on patch

            outputs = model.forward(patch.to(device=device)).cpu()


            # Extract predicted class and probability

            maxprob, predicted = outputs.max(dim=1)

            C_curr = predicted.item()

            P_curr = np.exp(maxprob.item())  # This is of the most probable class

            class_mat[j,i] = C_curr

            probs_mat[j,i] = (1-C_curr) * (1-P_curr) + C_curr*P_curr

            # This probability is for class 1 - water


            # Update count of overlaps for each pixel

            overlaps[y_curr:(y_curr+patch_size) , x_curr:(x_curr+patch_size)] += 1

            
            # Also update full classification and probability maps 

            # (from overlapping patches) of class 1

            class_mat_full[y_curr:(y_curr+patch_size) , x_curr:(x_curr+patch_size)] += class_mat[j,i]

            probs_mat_full[y_curr:(y_curr+patch_size) , x_curr:(x_curr+patch_size)] += probs_mat[j,i]

             

        time_elapsed = time.time() - since

        print('Row {:d} of {:d} per row {:.0f}m {:.0f}s'.format(i+1, (n_patches_x-1), time_elapsed // 60, time_elapsed % 60))

    
    # Scale by the number of overlaps for each pixel, so the the number 

    # of classifications and average probability per pixel is in [0,1]

    class_mat_full[overlaps>0] =  class_mat_full[overlaps>0] / overlaps[overlaps>0]

    probs_mat_full[overlaps>0] =  probs_mat_full[overlaps>0] / overlaps[overlaps>0]

    
    # Set those not covered by any patch to nan

    x_max_valid = (n_patches_x-1)*stride_overlap+patch_size - 1

    y_max_valid = (n_patches_y-1)*stride_overlap+patch_size - 1

    class_mat_full[(y_max_valid+1):, :] = np.nan

    class_mat_full[:, (x_max_valid+1):] = np.nan

    probs_mat_full[(y_max_valid+1):, :] = np.nan

    probs_mat_full[:, (x_max_valid+1):] = np.nan

    
    # Make map of uncertainty based on how many times the majority-vote

    # class is selected for each pixel

    uncertainty_overlap = np.zeros(class_mat_full.shape)

    uncertainty_overlap[class_mat_full > 0.5] = 1 - class_mat_full[class_mat_full>0.5]

    uncertainty_overlap[class_mat_full < 0.5] =     class_mat_full[class_mat_full<0.5]

    uncertainty_overlap = uncertainty_overlap * n_classes



    return class_mat_full, probs_mat_full, overlaps, uncertainty_overlap
