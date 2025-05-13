# ---- This is <inference_model.py> ----

"""
CNN for binary classification.

Trained to classify Sentinel-1 EW GRDM images at pixel level into the categories water (0) or ice (255).
Network takes 3 channels as input: sigma0 for HH, sigma0 for HV and incidence angle (IA).

Developed by Qiang Wang.
Code adapted, re-structed and packaged by Clément Stouls & Catherine Taelman.
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from time import time
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import glob
from loguru import logger
import sys
import json
import tifffile as tf
from tqdm import tqdm
from collections import OrderedDict 
import concurrent.futures
from UiT_asimoc.tools import create_datacube
from UiT_asimoc.tools import fill_nan_with_nearest
from UiT_asimoc.tools import Uncertainty_overlapping_patches
from UiT_asimoc.tools import UNet_3
from UiT_asimoc.build_class import inference_ice_water
import gc
import multiprocessing
from torch.cuda.amp import autocast

from scipy.ndimage import gaussian_filter

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Function to process a batch of data
def process_batch(patches, coords, model, device, img_dims, patch_size):
    class_sums = np.zeros(img_dims, dtype=np.float32)
    probs_sums = np.zeros(img_dims, dtype=np.float32)
    overlaps = np.zeros(img_dims, dtype=np.int32)

    patches = patches.to(device)
    with torch.no_grad():
        with autocast(): 
            outputs = model(patches)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, dim=1)
            probs_max = probs.max(dim=1)[0]  # Extract the maximum probability

        for i in range(len(coords[0])):
            x, y = coords[0][i].item(), coords[1][i].item()
            x, y = int(x), int(y)
            y_end = min(y + patch_size, img_dims[0])
            x_end = min(x + patch_size, img_dims[1])
            valid_patch_height = y_end - y
            valid_patch_width = x_end - x

            C_curr = predicted[i].cpu().numpy()[:valid_patch_height, :valid_patch_width]
            P_curr = probs_max[i].cpu().numpy()[:valid_patch_height, :valid_patch_width]
            probs_calc = (1 - C_curr) * (1 - P_curr) + C_curr * P_curr  

            class_sums[y:y_end, x:x_end] += C_curr
            probs_sums[y:y_end, x:x_end] += probs_calc
            overlaps[y:y_end, x:x_end] += 1

    return class_sums, probs_sums, overlaps

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

# Function to truncate arrays
def truncate_array(array, target_shape):
    """
    Truncates the array to the target shape.
    
    Parameters:
        array (np.ndarray): The input array.
        target_shape (tuple): The target shape.
    
    Returns:
        np.ndarray: The truncated array.
    """
    truncated_array = array[:target_shape[0], :target_shape[1]]
    return truncated_array

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #


def inference_model_uncertainty(model_name, path_to_model_folder, path_to_features, path_to_results, landmask=False, fill_nan=False, overwrite=False):
    start_time = time()

    loglevel = 'INFO'
    
    # Remove default logger handler and add personal one   
    logger.remove()
    logger.add(sys.stderr, level=loglevel)
    
    # Convert folder/file strings to paths   
    path_to_model_folder = Path(path_to_model_folder).expanduser().absolute()
    feat_folder          = Path(path_to_features).expanduser().absolute()
    results_folder       = Path(path_to_results).expanduser().absolute()
    
    if not path_to_model_folder.is_dir() or not feat_folder.is_dir() or not results_folder.is_dir():
        raise FileNotFoundError("Required directory not found")
    
    S1_id = feat_folder.stem
    
    model_json_filename = f'{model_name}_inference_config.json'
    ##model_json_filename = f'new_config_{model_name}.json'

    with open(path_to_model_folder / model_json_filename) as f: 
        model_json = json.load(f)
    model_path = model_json["pth_path"]
    model_name_without_extension = model_path.split('/')[-1][:-4]

    patch_height     = model_json["patch_height"]
    patch_width      = model_json["patch_width"]
    noise_correction = model_json["noise_correction"] 
    batch_size       = model_json["batch_size"]
    step_size        = model_json["step_size"]

    if step_size is None:
        raise FileNotFoundError('step_size must be specified in config file')

    HH_filename = f'Sigma0_HH_{noise_correction}_db.tif'
    HH_path     = feat_folder / HH_filename
    sigma0_HH   = tf.imread(HH_path.as_posix())

    HV_filename = f'Sigma0_HV_{noise_correction}_db.tif'
    HV_path     = feat_folder / HV_filename
    sigma0_HV   = tf.imread(HV_path.as_posix())

    IA_filename = 'IA.tif'
    IA_path     = feat_folder / IA_filename
    IA          = tf.imread(IA_path.as_posix())

    swath_mask_filename = 'swath_mask.tif'
    swath_mask_path     = feat_folder / swath_mask_filename
    swath_mask          = tf.imread(swath_mask_path.as_posix())

    if landmask:
        land_mask_filename = 'landmask.tif'
        land_mask_path     = feat_folder / land_mask_filename
        land_mask          = tf.imread(land_mask_path.as_posix())
        land_mask          = np.round(land_mask).astype(int)

    # Determine the target shape based on the smallest dimensions
    target_shape = (min(sigma0_HH.shape[0], IA.shape[0], swath_mask.shape[0]), min(sigma0_HH.shape[1], IA.shape[1], swath_mask.shape[1]))

    # Truncate arrays to the target shape
    sigma0_HH = truncate_array(sigma0_HH, target_shape)
    sigma0_HV = truncate_array(sigma0_HV, target_shape)
    IA        = truncate_array(IA, target_shape)
    swath_mask = truncate_array(swath_mask, target_shape)
    if landmask:
        land_mask = truncate_array(land_mask, target_shape)

    SAR_valid_mask = ~np.isnan(sigma0_HH)

    if fill_nan:
        sigma0_HH = fill_nan_with_nearest(sigma0_HH)
        sigma0_HV = fill_nan_with_nearest(sigma0_HV)

    if landmask:
        sigma0_HH[land_mask == 0] = np.nan
        sigma0_HV[land_mask == 0] = np.nan
        sigma0_HH = fill_nan_with_nearest(sigma0_HH)
        sigma0_HV = fill_nan_with_nearest(sigma0_HV)

    image_datacube, SAR_image_height, SAR_image_width = create_datacube(sigma0_HH, sigma0_HV, IA, patch_height, patch_width)
    img_dims = IA.shape

    del sigma0_HH, sigma0_HV, IA
    gc.collect()

    logger.info("Inference subimage with uncertainty estimation started")

    inference_tfm = transforms.Compose([
        transforms.Normalize((model_json["mean_channel_1"], model_json["mean_channel_2"], model_json["mean_channel_3"]),
                             (model_json["SD_channel_1"], model_json["SD_channel_2"], model_json["SD_channel_3"])),
    ])

    inference_set = inference_ice_water(image_datacube, patch_height, patch_width, step_size, transform=inference_tfm, return_coords=True)
    inference_loader = DataLoader(inference_set, batch_size=batch_size, shuffle=False, num_workers=8)  # Increased number of workers for parallel loading

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet_3(3, 2, bilinear=True)
    model_state = torch.load((path_to_model_folder / model_json["pth_path"]).as_posix(), map_location=device)

    if isinstance(model_state, OrderedDict):
        model.load_state_dict(model_state)
    else:
        model = model_state

    model.to(device)
    model.eval()

    class_sums = np.zeros(img_dims, dtype=np.float32)
    probs_sums = np.zeros(img_dims, dtype=np.float32)
    overlaps = np.zeros(img_dims, dtype=np.int32)

    with torch.no_grad():
        for patches, _, coords in tqdm(inference_loader, desc="Processing patches"):
            class_sum, probs_sum, overlap = process_batch(patches, coords, model, device, img_dims, patch_height)
            class_sums += class_sum
            probs_sums += probs_sum
            overlaps += overlap

    class_mat_full = np.divide(class_sums, overlaps, where=overlaps > 0, out=np.zeros_like(class_sums))
    probs_mat_full = np.divide(probs_sums, overlaps, where=overlaps > 0, out=np.zeros_like(probs_sums))

    uncertainty_overlap = np.zeros_like(class_mat_full)
    uncertainty_overlap[class_mat_full > 0.5] = 1 - class_mat_full[class_mat_full > 0.5]
    uncertainty_overlap[class_mat_full <= 0.5] = class_mat_full[class_mat_full <= 0.5]
    uncertainty_overlap = uncertainty_overlap * 2

    soft_uncertainty_overlap = np.zeros_like(class_mat_full)
    soft_uncertainty_overlap[class_mat_full > 0.5] = probs_mat_full[class_mat_full > 0.5]
    soft_uncertainty_overlap[class_mat_full <= 0.5] = 1 - probs_mat_full[class_mat_full <= 0.5]
    soft_uncertainty_overlap = soft_uncertainty_overlap * 2

    probs_no_ice = 1 - probs_mat_full
    epsilon = np.finfo(float).eps

    shannon_entropy = -(probs_mat_full * np.log2(probs_mat_full + epsilon) +
                        probs_no_ice * np.log2(probs_no_ice + epsilon))

    classified_image_masked = np.where((swath_mask > 0) & SAR_valid_mask, class_mat_full, np.nan)
    overlap_count_masked = np.where((swath_mask > 0) & SAR_valid_mask, overlaps, np.nan)
    Uncert_mat_full_maked = np.where((swath_mask > 0) & SAR_valid_mask, uncertainty_overlap, np.nan)
    probs_mat_full_masked = np.where((swath_mask > 0) & SAR_valid_mask, probs_mat_full, np.nan)
    shannon_entropy_masked = np.where((swath_mask > 0) & SAR_valid_mask, shannon_entropy, np.nan)
    soft_Uncert_mat_full_maked = np.where((swath_mask > 0) & SAR_valid_mask, soft_uncertainty_overlap, np.nan)

    if landmask:
        classified_image_masked = np.where(land_mask == 1, classified_image_masked, np.nan)
        overlap_count_masked = np.where(land_mask == 1, overlap_count_masked, np.nan)
        Uncert_mat_full_maked = np.where(land_mask == 1, Uncert_mat_full_maked, np.nan)
        probs_mat_full_masked = np.where(land_mask == 1, probs_mat_full_masked, np.nan)
        shannon_entropy_masked = np.where(land_mask == 1, shannon_entropy_masked, np.nan)
        soft_Uncert_mat_full_maked = np.where(land_mask == 1, soft_Uncert_mat_full_maked, np.nan)

    end_time = time()
    logger.info("Inference subimage with uncertainty finished, time elapsed for patch inference: : %s seconds" % (end_time - start_time))

    colors = ['black', 'blue', 'gray']
    cmap = matplotlib.colors.ListedColormap(colors)
    plt.imshow(classified_image_masked, cmap='cividis', interpolation='none', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.savefig(f"{results_folder}/{S1_id}_labels_{model_name_without_extension}.png", dpi=300)
    plt.close('all')

    plt.imshow(probs_mat_full_masked, cmap='cividis', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{results_folder}/{S1_id}_probability_{model_name_without_extension}.png", dpi=300)
    plt.close('all')

    plt.imshow(overlap_count_masked, cmap='cividis', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{results_folder}/{S1_id}_counts_{model_name_without_extension}.png", dpi=300)
    plt.close('all')

    plt.imshow(Uncert_mat_full_maked, cmap='cividis', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{results_folder}/{S1_id}_uncertainty_{model_name_without_extension}.png", dpi=300)
    plt.close('all')


    plt.imshow(soft_Uncert_mat_full_maked, cmap='cividis', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{results_folder}/{S1_id}_soft_uncertainty_{model_name_without_extension}.png", dpi=300)
    plt.close('all')

    plt.imshow(shannon_entropy_masked, cmap='cividis', interpolation='none')
    plt.colorbar()
    plt.savefig(f"{results_folder}/{S1_id}_shannon_{model_name_without_extension}.png", dpi=300)
    plt.close('all')

    # Save non-normalized label
    non_normalized_label = np.where((swath_mask > 0) & SAR_valid_mask, class_sums, np.nan)
    if landmask:
        non_normalized_label = np.where(land_mask == 1, non_normalized_label, np.nan)
    
    tf.imsave(f"{results_folder}/{S1_id}_non_normalized_labels_{model_name_without_extension}.tif", non_normalized_label.astype(np.float32))

    tf.imsave(f"{results_folder}/{S1_id}_labels_{model_name_without_extension}.tif", classified_image_masked.astype(np.float32))
    tf.imsave(f"{results_folder}/{S1_id}_probability_{model_name_without_extension}.tif", probs_mat_full_masked.astype(np.float32))
    tf.imsave(f"{results_folder}/{S1_id}_counts_{model_name_without_extension}.tif", overlap_count_masked.astype(np.float32))
    tf.imsave(f"{results_folder}/{S1_id}_uncertainty_{model_name_without_extension}.tif", Uncert_mat_full_maked.astype(np.float32))
    tf.imsave(f"{results_folder}/{S1_id}_shannon_{model_name_without_extension}.tif", shannon_entropy_masked.astype(np.float32))
    tf.imsave(f"{results_folder}/{S1_id}_soft_uncertainty_{model_name_without_extension}.tif", soft_Uncert_mat_full_maked.astype(np.float32))

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <inference_model.py> ----
