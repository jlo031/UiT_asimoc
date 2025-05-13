import json
import importlib.util
from UiT_asimoc.inference_model import inference_model_uncertainty

##from UiT_asimoc.write_raster_to_tif import write_to_tif

from pathlib import Path
from loguru import logger
import sys
import glob
import torch
import torch.multiprocessing as mp
import warnings

def check_directories_exist(directories):
    for directory in directories:
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return False
    return True

def process_file(model, path_to_model_folder, feat_folder, results_folder):
    try:
        # Add debug prints for folder paths
        logger.info(f"Model folder: {path_to_model_folder}")
        logger.info(f"Feature folder: {feat_folder}")
        logger.info(f"Results folder: {results_folder}")

        # Assuming that inference_model_uncertainty is where the error occurs
        inference_model_uncertainty(model, path_to_model_folder.as_posix(), feat_folder.as_posix(), results_folder.as_posix(), overwrite=True, landmask=True, fill_nan=True)
    
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        
        # Add debug statements to log array shapes
        logger.debug(f"Model: {model}")
        logger.debug(f"Feature folder contents: {list(feat_folder.glob('*'))}")
        logger.debug(f"Results folder contents: {list(results_folder.glob('*'))}")

def main():

    f = 'S1A_EW_GRDM_1SDH_20211230T072200_20211230T072300_041235_04E685_36C7'

    ML_level = [5,]

    loglevel = 'INFO'

    # Remove default logger handler and add personal one   
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    # Define the basic data directory for your project
    DATA_DIR = Path('/home/jo/work/CNN_TESTS/DATA/Sentinel-1/TestData/ml5x5')

    ##multilooked_features = BASE_DIR / 'Data'
    ##multilooked_features = BASE_DIR

    # Define path to model folder and model config json file
    path_to_model_folder = Path('/home/jo/work/UiT_asimoc/src/UiT_asimoc/models')

    model = 'jozef_NERSC_ml5x5'
    print(model)
        
    config_json_file = open(path_to_model_folder / model / f'{model}_inference_config.json')
    config_json = json.load(config_json_file)

    # Import model class as: from name_model import *, you can directly import your model class if you want
    loader = importlib.machinery.SourceFileLoader(config_json['model'], (path_to_model_folder / model / (config_json['model'] + ".py")).as_posix())
    spec = importlib.util.spec_from_loader(loader.name, loader)
    my_module = importlib.util.module_from_spec(spec)
    loader.exec_module(my_module)
        
    if "__all__" in my_module.__dict__:
        names = my_module.__dict__['__all__']
    else:
        names = [x for x in my_module.__dict__ if not x.startswith("_")]
    # Update globals
    globals().update({k: getattr(my_module, k) for k in names})
        


    f = DATA_DIR / f

    S1_name = Path(f).name
 
    logger.info(f'Processing {S1_name}')


    feat_folder = f

    logger.info(f'Feat folder: {feat_folder}')

    results_folder = DATA_DIR / f'results/{model}/{S1_name}/'

    logger.info(f'Results folder: {results_folder}')   

    results_folder.mkdir(parents=True, exist_ok=True)

    process_file(model, path_to_model_folder/model, feat_folder, results_folder)

if __name__ == "__main__":
    main()
