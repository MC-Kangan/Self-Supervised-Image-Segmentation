import numpy as np
import random
import os
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import Subset
import segmentation_models_pytorch as smp
from fine_tune_utils import *

import time
import json




class OxfordPetsDataset(Dataset):
    
    """
    A custom dataset class for the Oxford-IIIT Pet Dataset that handles loading of images and their corresponding masks,
    with optional transformations applied to both. 

    Attributes:
    - images_dir (str): Directory path containing the images.
    - masks_dir (str): Directory path containing the segmentation masks.
    - transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
      Applied to the images.
    - mask_transform (callable, optional): A function/transform that takes in a mask and returns a transformed version.
      Applied to the masks. If None, the `transform` specified will be applied to masks as well.

    The class expects the image and mask file names to match and be sorted in a way that each image corresponds to its mask.

    Methods:
    - __init__(self, images_dir, masks_dir, transform=None, mask_transform=None): Initializes the dataset with the directory
      paths and transformations.
    - __len__(self): Returns the total number of samples in the dataset.
    - preprocess_mask(self, mask): Preprocesses the mask by setting specific pixel values to denote background,
      foreground, and possibly edges or other classes.
    - __getitem__(self, idx): Fetches the image and its corresponding mask at the given index `idx`, applying any
      transformations specified to both.
    """
    
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform is not None else transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0 # Background
        # mask[(mask == 1.0)] = 0.5 # Item
        mask[(mask == 3.0) | (mask == 1.0)] = 1.0 # Edge
        return mask

    # It defines how an item from the dataset is fetched given its index
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        trimap = np.array(Image.open(mask_path))
        mask = self.preprocess_mask(trimap)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
    
    
def print_requires_grad_status(model):
    
    """
    Prints the requires_grad status of all parameters in a PyTorch model. This function iterates
    through each parameter in the model, printing its name along with whether it requires gradients
    to be calculated (requires_grad=True) or not (requires_grad=False). 
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model whose parameters' requires_grad status you want to check.

    Returns:
    - None: This function prints the status of each parameter to the console and does not return any value.
    """
    for name, param in model.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")
        
        
def train_segmentation_model(model, train_loader, val_loader, save_path = 'models', num_epochs=10, loss_type='dice', learning_rate=1e-4, device='cpu', model_name = 'model'):


    print(f'Training {model_name}')

    start_time = time.time()
    # Set the model to training mode and move it to the specified device
    model.train()
    model.to(device)

    # Define the optimizer - only parameters with requires_grad=True will be updated
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Define the loss function
    if loss_type == 'dice':
      criterion = smp.losses.DiceLoss(mode = 'binary')
    elif loss_type == 'docal':
      criterion = smp.losses.FocalLoss(mode = 'binary')
    else:
      # Binary Cross Entropy Loss With Sigmoid Function
      criterion = nn.BCEWithLogitsLoss()

    # Monitor loss in validation set
    dice_loss_fn = smp.losses.DiceLoss(mode='binary')
    focal_loss_fn = smp.losses.FocalLoss(mode='binary')
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # List to store loss values per batch
    batch_loss_list = []
    iou_scores_list, f1_scores_micro_list, f1_scores_macro_list, dice_loss_list, focal_loss_list, bce_loss_list = [], [], [], [], [], []

    best_iou = 0

    for epoch in range(num_epochs):
        running_loss_batch = 0.0
        running_loss = 0.0

        batch_loss_values = []

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)

            # Ensure masks are LongTensor and on the correct device
            masks = masks.float().to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Compute loss per every batch and cummulative loss
            running_loss_batch += loss.item()
            running_loss += loss.item()

            batch_loss_values.append(loss.item())

            if (i+1) % 10 == 0 :    # Print every 10 mini-batches
                print(f'[%d, %5d] {loss_type} loss: %.3f' %
                    (epoch + 1, i + 1, running_loss_batch / 10))
                running_loss_batch = 0

        train_loss = running_loss / (len(train_loader) * train_loader.batch_size) # Total loss / total number of training data
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}')
        batch_loss_list.append(batch_loss_values)

        if (epoch + 1)%2 ==0:
          model.eval()
          iou_scores, f1_scores_micro, f1_scores_macro, dice_loss, focal_loss, bce_loss = [], [], [], [], [], []

          with torch.no_grad():
              # Compute metrics on every batch
              for images, masks in val_loader:
                  images, masks = images.to(device), masks.float().to(device)
                  outputs = model(images)

                  # Compute SMP metrics
                  tp, fp, fn, tn = smp.metrics.get_stats(masks.long(), outputs.long(), mode='binary', threshold=0.5)

                  # For 'binary' case 'micro' = 'macro' = 'weighted'
                  iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                  f1_score_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                  f1_score_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

                  iou_scores.append(iou_score.item())
                  f1_scores_micro.append(f1_score_micro.item())
                  f1_scores_macro.append(f1_score_macro.item())
                  dice_loss.append(dice_loss_fn(outputs, masks).item())
                  focal_loss.append(focal_loss_fn(outputs, masks).item())
                  bce_loss.append(bce_loss_fn(outputs, masks).item())

          mean_iou = np.mean(iou_scores)
          mean_fscore_micro = np.mean(f1_scores_micro)
          mean_fscore_macro = np.mean(f1_scores_macro)
          mean_dice = np.mean(dice_loss)
          mean_focal = np.mean(focal_loss)
          mean_bce = np.mean(bce_loss)

          print(f'Validation (Epoch {epoch + 1}) - IoU: {mean_iou:.4f}, F1-score-micro: {mean_fscore_micro:.4f}, F1-score-macro: {mean_fscore_macro:.4f}, Dice-loss: {mean_dice:.4f}, Focal-loss: {mean_focal:.4f}, BCE-loss: {mean_bce:.4f}')

          iou_scores_list.append(mean_iou)
          f1_scores_micro_list.append(mean_fscore_micro)
          f1_scores_macro_list.append(mean_fscore_macro)
          dice_loss_list.append(mean_dice)
          focal_loss_list.append(mean_focal)
          bce_loss_list.append(mean_bce)

          if mean_iou > best_iou:
            # Example of saving a checkpoint after 10 epoch
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': batch_loss_values[-1],  # Last batch loss of the epoch
              }, f'{save_path}/{model_name}.pth')
            best_iou = mean_iou
            print(f'Epoch {epoch+1} checkpoint is saved.')

    # Save the final model
    torch.save(model.state_dict(), f'{save_path}/{model_name}_model.pth')

    print(f'Total training time: {time.time()-start_time:4f}s')

    metrics = {'batch_loss': batch_loss_list,
               'iou': iou_scores_list,
               'f1_micro': f1_scores_micro_list,
               'f1_macro': f1_scores_macro_list,
               'dice': dice_loss_list,
               'focal': focal_loss_list,
               'BCE': bce_loss_list}

    return model, metrics



if __name__ == "__main__":
    
    
    # IMPORTANT TRAINING SETTING 

    PRETRAIN_DATASET = 'cifar'
    LOSS_FUNC = 'BCE'
    DEV_SIZE = 20
    SAVE_PATH = 'models'
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

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
    
    if DEV_SIZE == 80:
        train_loader, test_loader, val_loader = data_split(dataset, ignore_size = 0)
    elif DEV_SIZE == 50:
        train_loader, test_loader, val_loader = data_split(dataset, ignore_size = 0.3)
    elif DEV_SIZE == 20:
        train_loader, test_loader, val_loader = data_split(dataset, ignore_size = 0.6)


    model = smp.Unet(
    encoder_name="resnet50",  # choose encoder, use `resnet50` for ResNet50
    encoder_weights=None,     # use `None` to not load the pre-trained weights
    in_channels=3,            # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,                # model output channels (number of classes in your dataset)
    )


    if PRETRAIN_DATASET == 'pet':
    # Load the weights into the model
        model.load_state_dict(torch.load("pretrain_model/pet_simclr_backbone.ckpt", map_location=torch.device(device)), strict=False)
    elif PRETRAIN_DATASET == 'cifar':
        model.load_state_dict(torch.load("pretrain_model/cifar_simclr_backbone.ckpt", map_location=torch.device(device)), strict=False)
    else: # baseline
        pass
    
    # print_requires_grad_status(model)
    
    # Call the training function
    model_name = f'{PRETRAIN_DATASET}_{LOSS_FUNC}_{DEV_SIZE}'
    model, metrics = train_segmentation_model(model, train_loader, val_loader, num_epochs=60, loss_type=LOSS_FUNC, learning_rate=1e-4, device=device, model_name = model_name)

    print('Check metrics file')
    for key, value in metrics.items():
        print(key, len(value))

    filename = f'{SAVE_PATH}/{PRETRAIN_DATASET}_{LOSS_FUNC}_{DEV_SIZE}.json'

    # Writing the dictionary to a file in JSON format
    with open(filename, 'w') as f:
        json.dump(metrics, f)



