# Import the necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Import torch vision
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import os 
import re
from torch.utils.data import DataLoader
# Import resnet50 model from torchvision
from torchvision.models import resnet50
from torch.optim.optimizer import Optimizer, required
# Import the dataset from huggingface
# pip install datasets
from datasets import load_dataset, Dataset
from typing import Tuple


"""
Function for loading the dataset
"""
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
        # return index != 4376 
        return index not in [4376, 7237, 8915, 14892, 14907, 17021, 19523, 22462]
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

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
    rnd_color_jitter,
    rnd_gray])
    return color_distort
"""
Function for training the model (loss function and optimizer)
"""

def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2  # 同一图片内部不同patch也是负样本
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)  # 同一图片内部不同patch也是负样本，除了自己和自己
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device = device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction = 'sum')
    loss /= n
    return loss  

class SimCLR(nn.Module):
    def __init__(self, model, temperature=0.1):
        super(SimCLR, self).__init__()
        # get the device of the model
        self.model = model
        # This is the two-layer MLP projection head as described in the paper whcih represents the g(.) function
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.temperature = temperature
        # Define the cosine similarity function
        # cosine_similarity = lambda z_i, z_j: torch.dot(z_i, z_j) / (torch.norm(z_i) * torch.norm(z_j))
    def forward(self, x_i, x_j):
        h_i = self.model(x_i)
        h_j = self.model(x_j)
        # print(h.shape)
        z_i = self.projection_head(h_i)
        # print(z_i.shape)
        z_j = self.projection_head(h_j)

        # Loss calculation by nt_xent_loss function
        loss = nt_xent_loss(z_i, z_j, self.temperature)
        return loss      
    


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=1e-6,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=0.001,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    # device = g_norm.get_device()
                    device = g_norm.device
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

if __name__ == '__main__':
    # Create directory to save the model
    directory_name = 'models_pets'
    os.makedirs(directory_name, exist_ok=True)
    # Start the training loop for SimCLR
    model = resnet50(pretrained=False) 
    model.fc = nn.Identity()
    simclr_model = SimCLR(model)

    # Hyperparameters
    accumulation_steps = 8  # For example, accumulate gradients over 8 steps before updating model weights
    batch_size = 120
    learning_rate = 0.3
    num_epochs = 20

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # this is the model you want to save for pre-training, where f(.) is the ResNet-50
    simclr_model = simclr_model.to(device)

    # define the dataset with and without data augmentation and with
    dataset = preprocess(load_dataset("cats_vs_dogs")['train'], device = device)
    dataset = SimClrData(huggingface_dataset=dataset)
    # split the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [18720, len(dataset) - 18720])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

    # Define the optimizer
    optimizer = LARS(
        [params for params in simclr_model.parameters() if params.requires_grad],
        lr=learning_rate,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )

    # Start the training loop for SimCLR
    mean_loss = 0
    best_loss = float('inf')
    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Initialize gradient accumulation
        loss_list = []
        # Set the model to training mode
        simclr_model.train()
        accumulation_loss = []
        for step, data in enumerate(train_loader):
            images_i = data[0].to(device)
            images_j = data[1].to(device)
            
            loss = simclr_model(images_i, images_j) / accumulation_steps  # Normalize the loss by accumulation steps
            loss.backward()  # Accumulate gradients
            loss_list.append(loss.item())
            accumulation_loss.append(loss.item())
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()  # Update model weights
                optimizer.zero_grad()  # Reset gradients
                # print the average loss of the accumulation steps
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {np.mean(accumulation_loss)}')
                accumulation_loss = []
        epoch_train_loss.append(np.mean(loss_list))
        # clear the cache
        # torch.cuda.empty_cache()
        # Validation loss
        # Set the model to evaluation mode
        simclr_model.eval()
        val_loss_list = []
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                images_i = data[0].to(device)
                images_j = data[1].to(device)
                loss = simclr_model(images_i, images_j)
                val_loss_list.append(loss.item())
            epoch_val_loss.append(np.mean(val_loss_list))
        # Save the model after each epoch name with the epoch number
        torch.save(model.state_dict(), f'{directory_name}/simclr_backbone_{epoch}.ckpt')
        torch.save(simclr_model.state_dict(), f'{directory_name}/simclr_model_{epoch}.ckpt')
        # Get the mean loss for the epoch
        val_loss = np.mean(val_loss_list)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss}')
        # save the best model
        if val_loss < best_loss:
            torch.save(model.state_dict(), f'{directory_name}/best_simclr_backbone.ckpt')
            torch.save(simclr_model.state_dict(), f'{directory_name}/best_simclr_model.ckpt')
            best_loss = val_loss
    
    # save epoch train and validation loss in json file
    with open(f'{directory_name}/epoch_train_loss.json', 'w') as f:
        json.dump(epoch_train_loss, f)
    with open(f'{directory_name}/epoch_val_loss.json', 'w') as f:
        json.dump(epoch_val_loss, f)