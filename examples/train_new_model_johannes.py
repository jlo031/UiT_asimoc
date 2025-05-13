import json
from ice_water_cnn_qiang.training_model import train_model
from pathlib import Path

# Define the basic data directory for your project
BASE_DIR = Path('/Users/tdo001/Box Sync/Jozef/JozefsCNN')
BASE_DIR = Path('/home/jlo031/CNN_TESTS/DATA/Sentinel-1')

# Define which model to train, and specify path to model folder that contains json config file
model = 'johannes_model_training_parameters'
path_to_model_folder = Path('/Users/tdo001/Box Sync/Jozef/JozefsCNN/ice_water_cnn_qiang/src/ice_water_cnn_qiang/model/')
path_to_model_folder = Path('/home/jlo031/CNN_TESTS/jozef/ice_water_cnn_qiang/src/ice_water_cnn_qiang/model/')

# Load config json file
config_json_path = path_to_model_folder / f'{model}.json'
with open(config_json_path, 'r') as config_json_file:
    config_json = json.load(config_json_file)

print("Configuration for training:")
print(json.dumps(config_json, indent=4))

# Define path to training data and results directory
training_data_folder = BASE_DIR / 'TrainingData' / 'ml5x5'
validation_data_folder = BASE_DIR / 'ValidationData' / 'ml5x5'
results_folder = BASE_DIR / 'Results' / 'new_model_johannes' / '100_adam_step_gamma1_0406_nt_0.0001wd1e-3p5cd0f05p5thdou030bn030'
#last_run = results_folder / 'UNET_last_epoch_model_NERSC_257.pth'
#last_run_report = results_folder / 'UNET_last_epoch_model_NERSC_257_report.pth'

# Ensure the directories exist
if not training_data_folder.exists():
    raise FileNotFoundError(f"Training data folder not found: {training_data_folder}")
if not validation_data_folder.exists():
    raise FileNotFoundError(f"Validation data folder not found: {validation_data_folder}")
if not path_to_model_folder.exists():
    raise FileNotFoundError(f"Model folder not found: {path_to_model_folder}")
if not results_folder.exists():
    results_folder.mkdir(parents=True, exist_ok=True)

# Start training
#train_model(model, path_to_model_folder.as_posix(), training_data_folder.as_posix(), validation_data_folder.as_posix(),
             #results_folder.as_posix(), restart=True, checkpoint_path=last_run.as_posix(), checkpoint_report_path=last_run_report.as_posix())
train_model(model, path_to_model_folder.as_posix(), training_data_folder.as_posix(), validation_data_folder.as_posix(),
             results_folder.as_posix(), restart=False)
