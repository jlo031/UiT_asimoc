# ---- This is <training_model.py> ----

"""
Training for binary ice-water classification
"""

import json
import sys
from loguru import logger
import pathlib
from time import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from UiT_asimoc.tools import UNet_3, eval_semantic_segmentation, FocalLoss_Ori, same_seeds
from UiT_asimoc.build_class import train_validate_ice_water

## Old imports that Jozef had but are not called anymore
##sys.path.append('/home/jozefjr/software/ice_water_cnn_qiang/src/ice_water_cnn_qiang/')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def train_model(model: str, path_to_model_folder: str, path_to_training_data: str, path_to_validation_data: str, path_to_results: str, restart: bool = False, checkpoint_path: str = None, checkpoint_report_path: str = None):
    loglevel = 'INFO'
    logger.remove()  
    logger.add(sys.stderr, level=loglevel)
    
    model_folder           = pathlib.Path(path_to_model_folder).expanduser().absolute()
    training_data_folder   = pathlib.Path(path_to_training_data).expanduser().absolute()
    validation_data_folder = pathlib.Path(path_to_validation_data).expanduser().absolute()
    results_folder         = pathlib.Path(path_to_results).expanduser().absolute()
    
    logger.debug(f'model_folder:           {path_to_model_folder}')
    logger.debug(f'training_data_folder:   {path_to_training_data}')
    logger.debug(f'validation_data_folder: {path_to_validation_data}')
    logger.debug(f'results_folder:         {path_to_results}')
    
    if not model_folder.is_dir():
        logger.error(f'Cannot find model folder: {path_to_model_folder}')
        raise FileNotFoundError(f'Cannot find model folder: {path_to_model_folder}')    
    if not training_data_folder.is_dir():     
        logger.error(f'Cannot find training data folder: {path_to_training_data}')     
        raise FileNotFoundError(f'Cannot find training data folder: {path_to_training_data}')    
    if not validation_data_folder.is_dir():     
        logger.error(f'Cannot find validation data folder: {path_to_validation_data}')     
        raise FileNotFoundError(f'Cannot find validation data folder: {path_to_validation_data}') 
    if not results_folder.is_dir():
        results_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f'Created results folder: {path_to_results}')


    # define paths to model json files
    model_training_config_json_file  = model_folder / f'{model}_training_config.json'
    model_inference_config_json_file = model_folder / f'{model}_inference_config.json'

    # load model training config
    with open(model_training_config_json_file, 'r') as f:
        model_training_config = json.load(f)
    ##model_training_config_json_file = open(model_folder / (model + '.json'))
    ##config_json = json.load(config_json_file)

    # extract model parameters from json config file
    patch_height            = model_training_config["patch_height"]
    patch_width             = model_training_config["patch_width"]
    batch_size              = model_training_config["batch_size"]
    step_size               = model_training_config["step_size"]
    num_epochs              = model_training_config["num_epochs"]
    num_channels            = model_training_config["num_channels"]
    num_classes             = model_training_config["num_classes"]
    early_stopping_patience = model_training_config["early_stopping_patience"]
    noise_correction        = model_training_config["noise_correction"]
    
    logger.info(f'Training CNN for max {num_epochs} epochs.')
    logger.info(f'Training CNN with {noise_correction}-denoised HH and HV features.')
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        logger.info("CUDA is applied.")
    else:
        logger.info("CUDA is not available, using CPU.")

    ##model_config_file = model_folder / f'new_config_{model}.json'

    same_seeds(19530617)
    
    training_set = train_validate_ice_water('training', training_data_folder, model_inference_config_json_file, patch_height, patch_width, step_size, noise_correction, batch_size, augment=False)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory=True)

    validation_set = train_validate_ice_water('validation', validation_data_folder, model_inference_config_json_file, patch_height, patch_width, step_size, noise_correction, batch_size)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=30, pin_memory=True)  
    
    our_net = UNet_3(num_channels, num_classes, bilinear=True)
    our_net = our_net.to(device)
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Define class frequencies
    class_0_freq = 0.75  # 70% of the samples are class 0
    class_1_freq = 0.25  # 30% of the samples are class 1

    # Calculate class weights (inverse of frequency)
    class_weights = torch.tensor([1.0 / class_0_freq, 1.0 / class_1_freq], dtype=torch.float32)

    # Optional: Normalize the weights (not required but can be useful)
    class_weights = class_weights / class_weights.sum()

    # Move class weights to the same device as the model
    class_weights = class_weights.to(device)

    # Define CrossEntropyLoss with calculated weights
    #criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    criterion = FocalLoss_Ori(2, alpha=[0.4, 0.6], gamma=1, ignore_index=None, reduction='mean')

    optimizer = torch.optim.Adam(our_net.parameters(), lr=0.0001, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                        verbose=True, threshold=0.0002, threshold_mode='rel', 
                                                        cooldown=0, min_lr=1e-10, eps=1e-08)
    store_train_loss = []
    store_train_accs = []
    store_train_miou_s = []
    store_train_class_acc_s = []
        
    store_valid_loss = []
    store_valid_accs = []
    store_valid_miou_s = []
    store_valid_class_acc_s = []

    best_valid_accs = -np.inf
    best_valid_loss = np.inf
    best_model_state = None
    best_epoch = -1
    counter = 0

    start_epoch = 0
    if restart and checkpoint_path and checkpoint_report_path:
        # Load the model checkpoint
        checkpoint_model = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint_model, torch.nn.Module):
            our_net.load_state_dict(checkpoint_model.state_dict())
        else:
            our_net.load_state_dict(checkpoint_model)

        # Load the checkpoint report
        checkpoint_report = torch.load(checkpoint_report_path, map_location=device)
        start_epoch = checkpoint_report['epoch']
        best_valid_loss = checkpoint_report['valid_loss']
        best_valid_acc = checkpoint_report['valid_acc']
        best_valid_miou = checkpoint_report['valid_miou']
        best_class_accs = checkpoint_report['valid_class_accs']
        best_model_state = our_net.state_dict()
        
        logger.info(f'Restarting training from epoch {start_epoch} with best validation loss {best_valid_loss:.5f}, best validation accuracy {best_valid_acc:.5f}, best validation mIoU {best_valid_miou:.5f}, and best class accuracies {best_class_accs}')

        results_filepath = results_folder / f'training_results_{noise_correction}.npz'
        if results_filepath.exists():
            data = np.load(results_filepath)
            store_train_loss = data['train_losses'].tolist()
            store_valid_loss = data['valid_losses'].tolist()
            store_train_accs = data['train_accs'].tolist()
            store_valid_accs = data['valid_accs'].tolist()
            store_train_miou_s = data['train_miou'].tolist()
            store_valid_miou_s = data['valid_miou'].tolist()

    train_start_time = time()
    
    logger.info('----------------------- training started -----------------------')
    
    for epoch in range(start_epoch, num_epochs):
        our_net.train()
        
        train_loss = []
        train_accs = []
        train_miou_s = []
        train_class_acc_s = []
    
        for batch in tqdm(training_loader):
            imgs, labels = batch

            # Debugging: Check shapes and types of input and labels
            assert imgs.shape[1:] == (num_channels, patch_height, patch_width), f"Unexpected image shape: {imgs.shape}"
            assert labels.dtype == torch.long, f"Labels must be of type torch.long, but got {labels.dtype}"
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                imgs, labels = imgs.to(device), labels.to(device)

                logits = our_net(imgs)
                
                # Debugging: Check if logits contain NaN or infinity
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.error("NaN or infinity in logits during training.")
                    continue

                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_label = logits.max(dim=1)[1].data.cpu().numpy()
            true_label = labels.data.cpu().numpy()

            eval_metric = eval_semantic_segmentation(pred_label, true_label)
            train_acc = eval_metric['mean_class_accuracy']
            train_miou = eval_metric['miou']
            train_class_acc = eval_metric['class_accuracy']

            train_loss.append(loss.item())
            train_accs.append(train_acc)
            train_miou_s.append(train_miou)
            train_class_acc_s.append(train_class_acc)
    
        train_loss = sum(train_loss) / len(train_loss)
        train_accs = sum(train_accs) / len(train_accs)
        train_miou_s = sum(train_miou_s) / len(train_miou_s)
        train_class_acc_s = sum(train_class_acc_s) / len(train_class_acc_s)
    
        store_train_loss.append(train_loss)
        store_train_accs.append(train_accs)
        store_train_miou_s.append(train_miou_s)
        store_train_class_acc_s.append(np.nanmean(train_class_acc_s))

        logger.info(f'|Epoch|: {epoch}\n|Train loss|:{train_loss:.5f}\n|Train acc|:{train_accs:.5f}\n|Train Mean IU|:{train_miou_s:.5f}\n|Train class_acc|:{train_class_acc_s}')
    
        our_net.eval()
    
        valid_loss = []
        valid_accs = []
        valid_miou_s = []
        valid_class_acc_s = []
    
        for batch in validation_loader:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = our_net(imgs)

                    # Debugging: Check for NaN and infinity in validation logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.error("NaN or infinity in logits during validation.")
                        continue

                    loss = criterion(logits, labels)
                
                # Debugging: Check for NaN or negative loss values
                if loss.item() != loss.item() or loss.item() < 0:
                    logger.error(f"Invalid loss value: {loss.item()}")
                    continue

            pred_label = logits.max(dim=1)[1].data.cpu().numpy()
            true_label = labels.data.cpu().numpy()

            eval_metric = eval_semantic_segmentation(pred_label, true_label)
            valid_acc = eval_metric['mean_class_accuracy']
            valid_miou = eval_metric['miou']
            valid_class_acc = eval_metric['class_accuracy']

            valid_loss.append(loss.item())
            valid_accs.append(valid_acc)
            valid_miou_s.append(valid_miou)
            valid_class_acc_s.append(valid_class_acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accs = sum(valid_accs) / len(valid_accs)
        valid_miou_s = sum(valid_miou_s) / len(valid_miou_s)
        valid_class_acc_s = sum(valid_class_acc_s) / len(valid_class_acc_s)

        # Store results
        store_valid_loss.append(valid_loss)
        store_valid_accs.append(valid_accs)
        store_valid_miou_s.append(valid_miou_s)
        store_valid_class_acc_s.append(np.nanmean(valid_class_acc_s))

        logger.info(f'|Validation loss|:{valid_loss:.5f}\n|Validation acc|:{valid_accs:.5f}\n|Validation Mean IU|:{valid_miou_s:.5f}\n|Validation class_acc|:{valid_class_acc_s}')

        # Step the scheduler at the end of each epoch
        scheduler.step(valid_loss)

        # Save the best model state
        if valid_loss < best_valid_loss - 0.00001:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_model_state = our_net.state_dict()
            counter = 0
            path_to_save_model = results_folder / f'UNET_best_model_{noise_correction}_{epoch}.pth'
            torch.save(our_net.state_dict(), path_to_save_model)
            torch.save({
                'epoch': epoch,
                'valid_loss': valid_loss,
                'valid_acc': valid_accs,
                'valid_miou': valid_miou_s,
                'valid_class_accs': valid_class_acc_s,
                'model_state_dict': our_net.state_dict()
            }, results_folder / f'UNET_best_model_{noise_correction}_{epoch}_report.pth')
        else:
            counter += 1
        
        # Early stopping
        if early_stopping_patience is not None and counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered. Saving model from epoch {epoch} with validation accuracy {valid_loss:.4f}.")
            our_net.load_state_dict(best_model_state)

            final_epoch_model_path = results_folder / f'UNET_early_stop_model_{noise_correction}_{epoch}.pth'
            torch.save(our_net.state_dict(), final_epoch_model_path)

            final_epoch_model_report_path = results_folder / f'UNET_early_stop_model_{noise_correction}_{epoch}_report.pth'
            torch.save({
                'epoch': epoch,
                'valid_loss': valid_loss,
                'valid_acc': valid_accs,
                'valid_miou': valid_miou_s,
                'valid_class_accs': valid_class_acc_s,
                'model_state_dict': our_net.state_dict()
            }, final_epoch_model_report_path)
            break

        # Save the model at each epoch
        epoch_model_path = results_folder / f'UNET_epoch_{noise_correction}_{epoch}.pth'
        torch.save(our_net.state_dict(), epoch_model_path)

        results_filepath = results_folder / f'training_results_{noise_correction}.npz'
        np.savez(results_filepath, train_losses=store_train_loss, valid_losses=store_valid_loss,
                 train_accs=store_train_accs, valid_accs=store_valid_accs,
                 train_miou=store_train_miou_s, valid_miou=store_valid_miou_s)
    
        # Plotting loss, accuracy, and mIoU after each epoch
        plt.figure()
        loss_figure_filepath = results_folder / f'plot_loss_{noise_correction}.png'
        plt.scatter(np.arange(1, len(store_train_loss) + 1, dtype=int), store_train_loss, c='g', label="Training")
        plt.scatter(np.arange(1, len(store_valid_loss) + 1, dtype=int), store_valid_loss, c='r', label="Validation")
        plt.xticks(np.arange(1, len(store_train_accs) + 1, 5))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(loss_figure_filepath.as_posix())
        plt.close()
        
        plt.figure()
        accs_figure_filepath = results_folder / f'plot_accuracies_{noise_correction}.png'
        plt.scatter(np.arange(1, len(store_train_accs) + 1, dtype=int), store_train_accs, c='g', label="Training")
        plt.scatter(np.arange(1, len(store_valid_accs) + 1, dtype=int), store_valid_accs, c='r', label="Validation")
        plt.xticks(np.arange(1, len(store_train_accs) + 1, 5))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(accs_figure_filepath.as_posix())
        plt.close()
    
        plt.figure()
        miou_figure_filepath = results_folder / f'plot_miou_{noise_correction}.png'
        plt.scatter(np.arange(1, len(store_train_miou_s) + 1, dtype=int), store_train_miou_s, c='g', label="Training")
        plt.scatter(np.arange(1, len(store_valid_miou_s) + 1, dtype=int), store_valid_miou_s, c='r', label="Validation")
        plt.xticks(np.arange(1, len(store_train_accs) + 1, 5))
        plt.xlabel('Epochs')
        plt.ylabel('Mean IoU')
        plt.legend()
        plt.savefig(miou_figure_filepath.as_posix())
        plt.close()
        
    train_end_time = time()
    
    logger.info("Training finished")
    logger.info("Running time for training process: %s seconds" % (train_end_time - train_start_time))
    
    if early_stopping_patience is not None and counter >= early_stopping_patience:
        logger.info(f"Early stopping triggered at epoch {epoch}. Validation accuracy: {valid_loss:.4f}.")
    else:
        logger.info(f"Training completed without early stopping at epoch {epoch}.")

    final_epoch_model_path = results_folder / f'UNET_last_epoch_model_{noise_correction}_{epoch}.pth'
    torch.save(our_net.state_dict(), final_epoch_model_path)

    final_epoch_model_report_path = results_folder / f'UNET_last_epoch_model_{noise_correction}_{epoch}_report.pth'
    torch.save({
        'epoch': epoch,
        'valid_loss': valid_loss,
        'valid_acc': valid_accs,
        'valid_miou': valid_miou_s,
        'valid_class_accs': valid_class_acc_s,
        'model_state_dict': our_net.state_dict()
    }, final_epoch_model_report_path)

    logger.info(f"Last epoch model and report saved at {final_epoch_model_path} and {final_epoch_model_report_path}.")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <training_model.py> ----
