import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import segmentation_models_pytorch as smp

import time
import json
import random

from fine_tune import *
from fine_tune_utils import *



def load_model(filepath):
    
    """
    Loads a PyTorch model from a specified file path. This function initializes a U-Net model with a ResNet50 encoder and then loads its weights from a given file.
    It can handle both cases where the model's state dictionary is directly saved or it is saved within a checkpoint
    dictionary that includes additional state information such as optimizer state.

    Args:
        filepath (str): The path to the file from which the model should be loaded. This file can either be a direct
                        state dictionary or a checkpoint containing the model's state dictionary among other information.

    Returns:
        torch.nn.Module: The loaded U-Net model, moved to the appropriate device (GPU if available, otherwise CPU).
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model= smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(filepath, map_location=torch.device(device))
    
    # Check if the loaded object is a checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from checkpoint format.")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        print("Model loaded from direct model state dictionary format.")
    else:
        raise ValueError("Unsupported file format")
    
    return model.to(device)


def visual_inspection(model, testset, inspect_num = 1, image_name = 'model_img'):
    
    """
    Perform visual inspection of model predictions against ground truth for a given test set image.

    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        testset (torch.utils.data.Dataset): The test dataset containing image-mask pairs.
        inspect_num (int, optional): The index of the image-mask pair in the test dataset to inspect.
                                     Default is 1.
        image_name (str, optional): Base name for the saved plot image file. Default is 'model_img'.

    Returns:
        None: This function does not return any value. It displays and saves a plot of the inspection results.
        
    """
    
    # Inspect performance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_img, test_mask = testset[inspect_num] 
    img, mask = test_img.to(device), test_mask.to(device)
    pred_mask = model(img.reshape(-1, 3, 224, 224))

    # Convert predictions to probability using sigmoid
    pred_prob = torch.sigmoid(pred_mask)
    
    # Convert probabilities to binary mask (e.g., threshold at 0.5)
    pred_binary_mask = (pred_prob > 0.5).float()

    # Move tensors to CPU for plotting
    img_cpu = img.squeeze().cpu().detach()
    mask_cpu = mask.squeeze().cpu().detach()
    pred_binary_mask_cpu = pred_binary_mask.squeeze().cpu().detach()

    # Normalize img for displaying purposes
    img_display = img_cpu.permute(1, 2, 0)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_display)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_cpu, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_binary_mask_cpu, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.savefig(f'results/{image_name}.png')
    plt.show()


def evaluate_model(model, test_loader, device, message):
    """
    Evaluates the performance of a segmentation model on a given test dataset, computing
    various metrics including IoU, F1 scores, Dice loss, Focal loss, BCE loss, recall, and accuracy.

    Args:
        model (torch.nn.Module): The trained segmentation model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device on which the model and data are to be loaded (e.g., 'cuda' or 'cpu').
        message (str): Custom message or identifier to print during evaluation and to name the saved image from visual inspection.

    Returns:
        tuple: A tuple containing the mean values of the following metrics:
               - IoU
               - F1-score (micro and macro)
               - Dice Loss
               - Focal Loss
               - BCE Loss
               - Recall
               - Accuracy
    """
    print(f'Evaluate: {message}')
    model.eval()

    # Monitor loss in validation set
    dice_loss_fn = smp.losses.DiceLoss(mode='binary')
    focal_loss_fn = smp.losses.FocalLoss(mode='binary')
    bce_loss_fn = nn.BCEWithLogitsLoss()
    iou_scores, f1_scores_micro, f1_scores_macro, dice_loss, focal_loss, bce_loss = [], [], [], [], [], []
    recalls, accuracys = [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.float().to(device)
            outputs = model(images)

            # Compute SMP metrics
            tp, fp, fn, tn = smp.metrics.get_stats(masks.long(), outputs.long(), mode='binary', threshold=0.5)

            # For 'binary' case 'micro' = 'macro' = 'weighted'
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f1_score_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

            iou_scores.append(iou_score.item())
            f1_scores_micro.append(f1_score_micro.item())
            f1_scores_macro.append(f1_score_macro.item())
            dice_loss.append(dice_loss_fn(outputs, masks).item())
            focal_loss.append(focal_loss_fn(outputs, masks).item())
            bce_loss.append(bce_loss_fn(outputs, masks).item())
            recalls.append(recall.item())
            accuracys.append(accuracy.item())

    mean_iou = np.mean(iou_scores)
    mean_fscore_micro = np.mean(f1_scores_micro)
    mean_fscore_macro = np.mean(f1_scores_macro)
    mean_dice = np.mean(dice_loss)
    mean_focal = np.mean(focal_loss)
    mean_bce = np.mean(bce_loss)
    mean_recall = np.mean(recalls)
    mean_accuracy = np.mean(accuracys)

    inspect_num = 25 # Choose a random number to inspect
    visual_inspection(model, test_loader.dataset, inspect_num, image_name=message)

    print(f'Testing - IoU: {mean_iou:.4f}, F1-score-micro: {mean_fscore_micro:.4f}, F1-score-macro: {mean_fscore_macro:.4f}, Dice-loss: {mean_dice:.4f}, Focal-loss: {mean_focal:.4f}, BCE-loss: {mean_bce:.4f}, Recall: {mean_recall:.4f}, Accuracy: {mean_accuracy:.4f}')
    print(' ')
    return (mean_iou, mean_fscore_micro, mean_fscore_macro, mean_dice, mean_focal, mean_bce, mean_recall, mean_accuracy)


if __name__ == "__main__":
    model_lst = sorted(os.listdir('models/model_files'))[1:]
    
    print(f'Models in the directory: {model_lst}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = OxfordPetsDataset(images_dir='images', masks_dir='annotations/trimaps', transform=transform, mask_transform=mask_transform)
    test_indices = load_indices('test_indices.txt')
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    
    model_result = []
    
    for i in range(len(model_lst)):
        model_name, loss_func, dev_split = model_lst[i].split('.')[0].split('_')
        dirname = os.path.join(f'models/model_files', model_lst[i])

        model = load_model(dirname)
        model.eval()
        mean_iou, mean_fscore_micro, mean_fscore_macro, mean_dice, mean_focal, mean_bce, mean_recall, mean_accuracy = evaluate_model(model, test_loader, device, message = model_lst[i].split('.')[0])

        model_result.append([model_name, loss_func, dev_split, mean_iou, mean_fscore_macro, mean_dice, mean_focal, mean_bce, mean_recall, mean_accuracy])
        
        
    df = pd.DataFrame(model_result, columns = ['model', 'loss_func', 'dev_split', 'iou', 'f1', 'dice', 'focal', 'BCE', 'recall', 'accuracy'])

    # df.to_csv('results/model_result.csv')