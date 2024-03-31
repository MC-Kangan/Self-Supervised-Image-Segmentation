import torch
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from typing import Tuple
from torchvision import transforms

# def set_seed(seed: int) -> None:
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
def preprocess(dataset: Dataset, device: torch.device) -> Dataset:
    """
    Preprocesses the input dataset by filtering out a specific example and setting the format to PyTorch tensors.

    Parameters:
    - dataset (Dataset): The Hugging Face dataset to be preprocessed.
    - device (torch.device): The device to which the dataset tensors should be moved (e.g., CPU or CUDA device).

    Returns:
    - Dataset: A dataset with the specified example removed and all data formatted as PyTorch tensors, ready for model training or inference.
    """
    def filter_out_example(example, index):
        return index != 4376 
    filtered_dataset = dataset.filter(filter_out_example, with_indices=True)
    filtered_dataset = filtered_dataset.with_format("torch", device = device)
    return filtered_dataset
    
class SimClrData(Dataset):
    """
    A PyTorch Dataset class for the SimCLR model, incorporating specific data augmentations as per the SimCLR framework.

    The class applies a series of transformations to each image in the input dataset to generate two augmented views
    of the same image, which are used as a positive pair for contrastive learning.

    Parameters:
    - huggingface_dataset (Dataset): A dataset loaded from Hugging Face's datasets library.

    Returns:
    - A SimClrData instance capable of iterating over transformed image pairs.
    """
    def __init__(self, huggingface_dataset: Dataset) -> None:
        self.dataset = huggingface_dataset
        
        image_height = 224
        image_width = 224

        def get_color_distortion(s=1.0):
            # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
            return color_distort
        # This is the best combination of data augmentation techniques for the SimCLR model shown in the paper
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),  # Convert tensor to PIL Image
            transforms.RandomResizedCrop((image_height, image_width)), # This follow the random cropping and resizing in the paper
            get_color_distortion(s=1),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)), # In the paper, the kernel size is 10% of the image height and width and sigma is between 0.1 and 2.0. As the kernel size must be odd, we choose 23 as the kernel size.
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # transform the dimension order for huggingface dataset
        image = self.dataset[index]['image'].permute(2,0,1)
        
        # apply transform and augmentation
        image_1 = self.data_transforms(image)
        image_2 = self.data_transforms(image)
        
        return image_1, image_2
        
        