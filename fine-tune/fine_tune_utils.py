import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset


def create_split_indices(dataset_size, test_split=0.2, random_seed=42):
    """
    Generates a list of shuffled indices for training and testing datasets based on the specified dataset size and test split ratio.

    Parameters:
    - dataset_size (int): The total number of samples in the dataset.
    - test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    - random_seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
    - tuple: A tuple containing two lists of indices, (train_indices, test_indices), representing the indices for the training and testing datasets, respectively.
    """
    
    # Setting the random seed for reproducibility
    torch.manual_seed(random_seed)

    # Generating shuffled indices
    indices = torch.randperm(dataset_size).tolist()

    # Calculating the split index
    split = int(len(indices) * (1 - test_split))

    # Splitting the indices
    train_indices, test_indices = indices[:split], indices[split:]

    return train_indices, test_indices


def get_dataset_subsets(dataset, train_indices, test_indices):
    
    """
    Creates two dataset subsets for training and testing from the given dataset and indices.

    Parameters:
    - dataset (Dataset): The original dataset to split into subsets.
    - train_indices (list): A list of indices for the training subset.
    - test_indices (list): A list of indices for the test subset.

    Returns:
    - tuple: A tuple containing two dataset subsets, (train_subset, test_subset), for training and testing, respectively.
    """

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset


def save_indices(filename, indices):
    
    """
    Saves the given list of indices to a file.

    Parameters:
    - filename (str): The name of the file to save the indices to.
    - indices (list): A list of indices to save.

    Returns:
    - None
    """
    
    # Open a file in write mode ('w') and use a context manager
    with open(filename, 'w') as f:
        for index in indices:
            # Write each index to the file, followed by a newline character
            f.write(f"{index}\n")

    print(f'{filename} is saved.')


def load_indices(filename):
    
    """
    Loads and returns a list of indices from a specified file.

    Parameters:
    - filename (str): The name of the file to load the indices from.

    Returns:
    - list: A list of indices loaded from the file.
    """
    
    # Open the file in read mode ('r') and use a context manager
    with open(filename, 'r') as f:
        # Read the lines, convert each line back to an integer, and remove newline characters
        loaded_indices = [int(line.strip()) for line in f]
    print(f'{filename} is loaded.')
    return loaded_indices


def data_split(dataset, ignore_size = 0):
    
    """
    The function first splits the dataset into development (dev) and test sets, then further splits
    the dev set into training and validation sets. An optional portion of the data can be ignored
    from the dev set based on the ignore_size parameter. The indices of each split are saved to
    text files, and DataLoader objects are created for training, validation, and test sets.

    Parameters:
    - dataset (Dataset): The dataset to split. Should be an object that supports indexing and len().
    - ignore_size (float, optional): The proportion of the dataset to ignore from the dev set. Defaults to 0, meaning no data is ignored.

    Returns:
    - tuple: A tuple containing DataLoader objects for the training, test, and validation sets, in that order.
    
    This function also prints the size of each dataset subset and their respective proportions of the total dataset.
    """
    
    # Default spliting: 20% testing data
    final_dev_size = 1 - 0.2 - ignore_size
    ignore_portion = 1 - (final_dev_size / 0.8)

    # Create split indices
    dev_indices, test_indices = create_split_indices(len(dataset), test_split=0.2, random_seed=42)

    dev_indices, ignore_indices = create_split_indices(len(dev_indices), test_split=ignore_portion, random_seed=42)

    # Create dataset subsets
    dev_set, test_set = get_dataset_subsets(dataset, dev_indices, test_indices)

    # Create split indices
    train_indices, val_indices = create_split_indices(len(dev_indices), test_split=0.1, random_seed=42)
    # Create dataset subsets
    train_set, val_set = get_dataset_subsets(dev_set, train_indices, val_indices)

    save_indices('train_indices.txt', train_indices)
    save_indices('val_indices.txt', val_indices)
    save_indices('test_indices.txt', test_indices)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

    print(f'Train set size: {len(train_set)}, Test set size: {len(test_set)}, Val set size: {len(val_set)}, Total data size: {len(dataset)}')
    print(f'Train set %: {len(train_set)/len(dataset) * 100}, Test set %: {len(test_set)/len(dataset) * 100}, Val set %: {len(val_set)/len(dataset) * 100}')

    return train_loader, test_loader, val_loader



if __name__ == "__main__":
    pass