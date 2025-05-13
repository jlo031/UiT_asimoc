from osgeo import gdal
import os
from pathlib import Path

# path to tif
BASE_DIR = Path('/scratch/cta014/Sentinel1/CNN_ice_water/inference/results')
#Path("/media/cirfa/CIRFA_media/CNN_training_Catherine/inference_results/S1A_EW_GRDM_1SDH_20201201T070428_20201201T070533_035489_04261E_5EA0")

results_dir_list = glob.glob((BASE_DIR / f'S1*').as_posix())

for S1_dir in results_dir_list:
    S1_dir = Path(S1_dir)
    s1_basename = S1_dir.stem
    print(f'Compressing results of {s1_basename}'

    # construct input and output filename
    in_filename = f'{s1_basename}_labels_UNET_trained_NERSC_50.tif'
    out_filename = in_filename.split('.')[0] + '_LZW.tif'
    
    # define paths of input and output files
    input_file = S1_dir / in_filename
    output_file = S1_dir / out_filename

    # compress input file
    compression_command = f"gdal_translate -co compress=LZW {input_file} {output_file}" 
    os.system(compression_command)
