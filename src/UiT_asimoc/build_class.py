# ---- This is <build_class.py> ----

"""
build_class module for ice-water classification with UNET
"""

import pathlib
import json
import glob
import warnings

from loguru import logger

import numpy as np

import rasterio
from rasterio.windows import Window

from skimage.util import view_as_windows
import gc
import concurrent.futures

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from UiT_asimoc.tools import same_seeds

## Old imports that Jozef had but are not called anymore
##import multiprocessing as mp
##from patchify import patchify
##import tifffile as tf

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class train_validate_ice_water(Dataset):
    def __init__(self, mode, feature_folder, model_config_file, patch_height, patch_width, step_size, noise_correction, batch_size, mean_HH=None, mean_HV=None, mean_IA=None, std_HH=None, std_HV=None, std_IA=None, normalize=True, augment=False):
        self.mode              = mode
        self.feature_folder    = pathlib.Path(feature_folder)
        self.model_config_file = model_config_file
        self.patch_height      = patch_height
        self.patch_width       = patch_width
        self.step_size         = step_size
        self.noise_correction  = noise_correction
        self.batch_size        = batch_size
        self.normalize         = normalize  # New flag to control normalization
        self.augment           = augment  

        self.S1_IDs = self.list_files_with_prefix(self.feature_folder, 'S1')


        # compute mean and standard deviation of features if not specified
        if self.mode == 'training' and (mean_HH is None or mean_HV is None or mean_IA is None or std_HH is None or std_HV is None or std_IA is None):
            self.mean_HH, self.mean_HV, self.mean_IA, self.std_HH, self.std_HV, self.std_IA = self.calculate_statistics()
            self.update_model_config()
        else:
            if mean_HH is not None:
                self.mean_HH = mean_HH
            if mean_HV is not None:
                self.mean_HV = mean_HV
            if mean_IA is not None:
                self.mean_IA = mean_IA
            if std_HH is not None:
                self.std_HH = std_HH
            if std_HV is not None:
                self.std_HV = std_HV
            if std_IA is not None:
                self.std_IA = std_IA
            self.load_model_config()

        same_seeds(19530617)

        if self.normalize:
            transforms_list = [ transforms.Normalize((self.mean_HH, self.mean_HV, self.mean_IA), (self.std_HH, self.std_HV, self.std_IA)) ]

            if self.augment and self.mode == 'training':
                # Add augmentation transforms to the list
                #transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))  # Horizontal flip with 50% chance
                transforms_list.append(transforms.RandomVerticalFlip(p=0.5))    # Vertical flip with 50% chance

            # Now create the final transform pipeline with both normalization and augmentations
            self.transform = transforms.Compose(transforms_list)
        else:
            # No normalization or augmentation
            self.transform = None

        self.valid_patches = self.create_valid_patch_index()

    # ------------------------ #

    @staticmethod
    def list_files_with_prefix(directory, prefix):
        path  = pathlib.Path(directory)
        items = [item.name for item in path.iterdir() if item.is_dir() and item.name.startswith(prefix)]
        return items

    # ------------------------ #

    def load_image_data_and_labels(self, feature_folder):
        HH_path     = feature_folder / f'Sigma0_HH_{self.noise_correction}_db.tif'
        HV_path     = feature_folder / f'Sigma0_HV_{self.noise_correction}_db.tif'
        IA_path     = feature_folder / 'IA.tif'
        labels_path = feature_folder / 'labels.tif'

        # load labels and find indices with non-zero labels
        with rasterio.open(labels_path) as src:
            label_mask = src.read(1)

            # Apply get_nonzero_window logic
            non_zero_indices = np.nonzero(label_mask)
            min_y, min_x     = np.min(non_zero_indices, axis=1)
            max_y, max_x     = np.max(non_zero_indices, axis=1)

            min_y = max(0, min_y + 5)
            min_x = max(0, min_x + 5)
            max_y = min(label_mask.shape[0], max_y - 5)
            max_x = min(label_mask.shape[1], max_x - 5)

            valid_window = Window(min_x, min_y, max_x - min_x, max_y - min_y)
            label_mask = src.read(1, window=valid_window)

        with rasterio.open(HH_path) as src:
            sigma0_HH = src.read(1, window=valid_window)

        with rasterio.open(HV_path) as src:
            sigma0_HV = src.read(1, window=valid_window)

        with rasterio.open(IA_path) as src:
            IA = src.read(1, window=valid_window)

        # create label_mask and data_cube
        label_mask = label_mask.astype(np.int64)
        image_datacube = np.stack((sigma0_HH, sigma0_HV, IA))

        del sigma0_HH, sigma0_HV, IA

        return image_datacube, label_mask

    # ------------------------ #

    def get_patch_window(self, idx, shape, min_x, min_y):
        num_patches_x = (shape[1] - self.patch_width) // self.step_size + 1
        num_patches_y = (shape[0] - self.patch_height) // self.step_size + 1

        patch_row = idx // num_patches_x
        patch_col = idx % num_patches_x

        start_x = patch_col * self.step_size + min_x
        start_y = patch_row * self.step_size + min_y

        return Window(start_x, start_y, self.patch_width, self.patch_height)

    # ------------------------ #

    def process_image(self, S1_ID):
        S1_feat_folder = self.feature_folder / f"{S1_ID}"
        image_datacube, label_mask = self.load_image_data_and_labels(S1_feat_folder)

        if image_datacube is None or label_mask is None:
            return None, None

        if image_datacube.shape[1] < self.patch_height or image_datacube.shape[2] < self.patch_width:
            logger.warning(f"Skipping {S1_ID} due to patch size ({self.patch_height}, {self.patch_width}) being too large for the image dimensions ({image_datacube.shape[1]}, {image_datacube.shape[2]}).")
            return None, None

        data_patches = self.create_patches(image_datacube, self.patch_height, self.patch_width, self.step_size)
        label_patches = self.create_patches(label_mask, self.patch_height, self.patch_width, self.step_size, is_label=True)

        nonzero_indices = [idx for idx in range(label_patches.shape[0])
                           if not 0 in label_patches[idx, :, :] and not np.isnan(data_patches[idx]).any()]
        data_patches_selection = data_patches[nonzero_indices, :, :, :]
        label_patches_selection = label_patches[nonzero_indices, :, :]

        # Map labels 1 and 2 to 0 and 1
        label_patches_selection -= 1

        return data_patches_selection, label_patches_selection

    # ------------------------ #

    def create_patches(self, array, patch_height, patch_width, step_size, is_label=False):
        if is_label:
            windows = view_as_windows(array, (patch_height, patch_width), step=step_size)
            return windows.reshape(-1, patch_height, patch_width)
        else:
            windows = view_as_windows(array, (array.shape[0], patch_height, patch_width), step=(1, step_size, step_size))
            return windows.reshape(-1, array.shape[0], patch_height, patch_width)

    # ------------------------ #

    def process_S1_ID(self, S1_ID):
        data_patches_selection, _ = self.process_image(S1_ID)
        if data_patches_selection is not None:
            num_pixels = data_patches_selection.shape[0] * self.patch_height * self.patch_width
            sum_HH = np.sum(data_patches_selection[:, 0, :, :])
            sum_HV = np.sum(data_patches_selection[:, 1, :, :])
            sum_IA = np.sum(data_patches_selection[:, 2, :, :])
            sum_HH_sq = np.sum(data_patches_selection[:, 0, :, :] ** 2)
            sum_HV_sq = np.sum(data_patches_selection[:, 1, :, :] ** 2)
            sum_IA_sq = np.sum(data_patches_selection[:, 2, :, :] ** 2)
            return sum_HH, sum_HV, sum_IA, sum_HH_sq, sum_HV_sq, sum_IA_sq, num_pixels
        return 0, 0, 0, 0, 0, 0, 0

    # ------------------------ #

    def calculate_statistics(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_S1_ID, self.S1_IDs))

        total_HH, total_HV, total_IA, total_HH_sq, total_HV_sq, total_IA_sq, total_patches = map(sum, zip(*results))

        mean_HH = total_HH / total_patches
        mean_HV = total_HV / total_patches
        mean_IA = total_IA / total_patches

        std_HH = np.sqrt(total_HH_sq / total_patches - mean_HH ** 2)
        std_HV = np.sqrt(total_HV_sq / total_patches - mean_HV ** 2)
        std_IA = np.sqrt(total_IA_sq / total_patches - mean_IA ** 2)

        logger.info(f'Means are: {mean_HH}, {mean_HV}, {mean_IA}')
        logger.info(f'STDs are: {std_HH}, {std_HV}, {std_IA}')

        return mean_HH, mean_HV, mean_IA, std_HH, std_HV, std_IA

    # ------------------------ #

    def update_model_config(self):
        try:
            with open(self.model_config_file, "r") as f:
                model_config = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.model_config_file}")
            raise

        model_config['mean_channel_1'] = np.around(float(self.mean_HH), 2)
        model_config['mean_channel_2'] = np.around(float(self.mean_HV), 2)
        model_config['mean_channel_3'] = np.around(float(self.mean_IA), 2)
        model_config['SD_channel_1'] = np.around(float(self.std_HH), 2)
        model_config['SD_channel_2'] = np.around(float(self.std_HV), 2)
        model_config['SD_channel_3'] = np.around(float(self.std_IA), 2)

        with open(self.model_config_file, "w") as f:
            json.dump(model_config, f, indent=4)

    # ------------------------ #

    def load_model_config(self):
        with open(self.model_config_file) as f:
            model_config = json.load(f)
        
        self.mean_HH = np.around(model_config['mean_channel_1'], 2)
        self.mean_HV = np.around(model_config['mean_channel_2'], 2)
        self.mean_IA = np.around(model_config['mean_channel_3'], 2)
        self.std_HH = np.around(model_config['SD_channel_1'], 2)
        self.std_HV = np.around(model_config['SD_channel_2'], 2)
        self.std_IA = np.around(model_config['SD_channel_3'], 2)

    # ------------------------ #

    def check_valid_patches(self, S1_ID):
        valid_patches = []
        S1_feat_folder = self.feature_folder / f"{S1_ID}"
        image_datacube, label_mask = self.load_image_data_and_labels(S1_feat_folder)

        num_patches_x = (image_datacube.shape[2] - self.patch_width) // self.step_size + 1
        num_patches_y = (image_datacube.shape[1] - self.patch_height) // self.step_size + 1
        total_patches = num_patches_x * num_patches_y

        for idx in range(total_patches):
            patch_row = idx // num_patches_x
            patch_col = idx % num_patches_x

            start_x = patch_col * self.step_size
            start_y = patch_row * self.step_size

            patch_data = image_datacube[:, start_y:start_y+self.patch_height, start_x:start_x+self.patch_width]
            patch_label = label_mask[start_y:start_y+self.patch_height, start_x:start_x+self.patch_width]

            if not np.any(patch_label == 0) and not np.isnan(patch_data).any():
                valid_patches.append((S1_ID, idx))

        return valid_patches

    # ------------------------ #

    def create_valid_patch_index(self):
        valid_patches = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self.check_valid_patches, self.S1_IDs))
            for result in results:
                valid_patches.extend(result)
        return valid_patches

    # ------------------------ #

    def __len__(self):
        return len(self.valid_patches)

    # ------------------------ #

    def __getitem__(self, index):
        S1_ID, patch_idx = self.valid_patches[index]
        S1_feat_folder = self.feature_folder / f"{S1_ID}"

        image_datacube, label_mask = self.load_image_data_and_labels(S1_feat_folder)
        
        num_patches_x = (image_datacube.shape[2] - self.patch_width) // self.step_size + 1
        num_patches_y = (image_datacube.shape[1] - self.patch_height) // self.step_size + 1

        patch_row = patch_idx // num_patches_x
        patch_col = patch_idx % num_patches_x

        start_x = patch_col * self.step_size
        start_y = patch_row * self.step_size

        patch_data = image_datacube[:, start_y:start_y+self.patch_height, start_x:start_x+self.patch_width]
        patch_label = label_mask[start_y:start_y+self.patch_height, start_x:start_x+self.patch_width]

        # Apply transformation
        if self.transform:
            patch_data = torch.tensor(patch_data, dtype=torch.float32)
            patch_data = self.transform(patch_data)

        patch_label = torch.tensor(patch_label, dtype=torch.long)

        # Map labels 1 and 2 to 0 and 1
        patch_label -= 1
        
        gc.collect()
        return patch_data, patch_label

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class inference_ice_water(Dataset):
    def __init__(self, image_datacube, patch_height, patch_width, step_size, transform=None, return_coords=False): 
        self.transform      = transform
        self.return_coords  = return_coords
        self.image_datacube = image_datacube
        self.patch_height   = patch_height
        self.patch_width    = patch_width
        self.step_size      = step_size

        self.datacube_depth, self.datacube_height, self.datacube_width = image_datacube.shape
        self.label_image = np.zeros((self.datacube_height, self.datacube_width))

        self.N_rows = (self.datacube_height - patch_height) // step_size + 1
        self.N_cols = (self.datacube_width - patch_width) // step_size + 1

        if self.return_coords:
            self.patch_coords = self.calculate_patch_coordinates((self.datacube_height, self.datacube_width), patch_height, patch_width, step_size)

    # ------------------------ #

    def calculate_patch_coordinates(self, image_dims, patch_height, patch_width, step_size):
        datacube_height, datacube_width = image_dims
        coords = [(x, y) for y in range(0, datacube_height - patch_height + 1, step_size) for x in range(0, datacube_width - patch_width + 1, step_size)]
        return coords

    # ------------------------ #

    def __len__(self):
        return self.N_rows * self.N_cols

    # ------------------------ #

    def __getitem__(self, index):
        row_idx = index // self.N_cols
        col_idx = index % self.N_cols

        y = row_idx * self.step_size
        x = col_idx * self.step_size

        data_patch  = self.image_datacube[:, y:y+self.patch_height, x:x+self.patch_width]
        label_patch = self.label_image[y:y+self.patch_height, x:x+self.patch_width]

        data_patch_tensor = torch.from_numpy(data_patch).float()
        label_patch_tensor = torch.from_numpy(label_patch).float()

        if self.transform:
            data_patch_tensor = self.transform(data_patch_tensor)

        if self.return_coords:
            coords = (x, y)
            return data_patch_tensor, label_patch_tensor, coords

        return data_patch_tensor, label_patch_tensor

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <build_class.py> ----
