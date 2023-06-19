import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from lightly.data.dataset import LightlyDataset
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from tqdm import tqdm
 
from autoSSL.utils import  dict2transformer
class PipeDataset():
    def __init__(
        self, input_dir=None, augmentation=None, samples=None, batch_size=1, shuffle=False, drop_last=False, num_workers=0, config=None
    ):
        if config is not None:
            # Extract the necessary parameters from the config dictionary
            config=config.copy()
            input_dir = config["dataset_dir"]
            view = config["view"]
            samples = config["samples"]
            batch_size = config["batch_size"]
            shuffle = config["shuffle"]
            drop_last = config["drop_last"]
            num_workers = config["num_workers"]
        
        self.transform = augmentation
        
        # Load the data using LightlyDataset with the specified transform
        self._dataset = LightlyDataset(input_dir=input_dir, transform=self.transform)

        # Set the number of samples to be loaded if specified
        if samples:
            self._dataset = self._random_subset(self.dataset, samples)

        # Initialize the DataLoader
        self._dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        self.dir = input_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    def split(self, ratio):
        """
        Split the dataset into two separate PipeDataset instances based on the given ratio.

        Parameters:
        ratio (float): The ratio of samples to include in the first split.

        Returns:
        dataset1 (PipeDataset): The first split of the dataset.
        dataset2 (PipeDataset): The second split of the dataset.
        """
        assert 0 <= ratio <= 1, "Ratio must be between 0 and 1"

        # Calculate the number of samples in each split
        total_samples = len(self.dataset)
        split1_samples = int(total_samples * ratio)
        split2_samples = total_samples - split1_samples

        # Create random indices for the splits
        indices = list(range(total_samples))
        random.shuffle(indices)
        split1_indices = indices[:split1_samples]
        split2_indices = indices[split1_samples:]

        # Create subsets for the splits
        subset1 = torch.utils.data.Subset(self.dataset, split1_indices)
        subset2 = torch.utils.data.Subset(self.dataset, split2_indices)

        # Create new PipeDataset instances for the splits
        dataset1 = PipeDataset(
            input_dir=self.dir,
            augmentation=None,
            samples=None,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )
        dataset2 = PipeDataset(
            input_dir=self.dir,
            augmentation=None,
            samples=None,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )

        # Set the subsets as the dataset for each split
        dataset1.update(subset1) 
        dataset2.update(subset2)

        return dataset1, dataset2

    
    def create_KNNtest(self):
        # Initialize the DataLoader
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )

    
    
    def update(self, dataset):
        """
        Update the current PipeDataset with a new dataset.

        Parameters:
        dataset (LightlyDataset): The new dataset to replace the current dataset.
        """
        self._dataset = dataset
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    
 
    def plot(self, indices):
        if not isinstance(indices, (list, tuple)):
                indices = list(indices)  # Convert range to list
        if isinstance(self.dataset[indices[0]][0], list):
            from lightly.utils.debug import plot_augmented_images
            import lightly
            input_images=[LightlyDataset(input_dir=self.dir)[index][0] for index in indices]
            collate_fn2 =lightly.data.collate.MultiViewCollateFunction([self.transform.aug for i in range(self.transform.nview)])   
            fig = plot_augmented_images(input_images, collate_fn2)
        else:
            rows = math.ceil(len(indices) / 3)
            fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))

            for i, index in enumerate(indices):
                img, label, title = self.dataset[index]

                # Rescale the image to the range [0, 1]
                img_min = img.min()
                img_max = img.max()
                img_rescaled = (img - img_min) / (img_max - img_min)

                if rows > 1:
                    ax = axes[i // 3, i % 3]
                else:
                    ax = axes[i % 3]
                ax.imshow(img_rescaled.permute(1, 2, 0))
                ax.set_title(f"{title}\nLabel: {label}", fontsize=8)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

    def _random_subset(self, dataset, samples):
        indices = random.sample(range(len(dataset)), samples)
        return torch.utils.data.Subset(dataset, indices)

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def array(self):
        return dataloader2array(self.dataloader)

    @property
    def dataset(self):
        return self._dataset

def dataloader2array(dataloader):
    """
    This function extracts data from the provided dataloader using the given model.

    Parameters:
    dataloader (DataLoader): The DataLoader object to iterate over.

    Returns:
    data_list (list): A list of numpy arrays, each array corresponding to a type of data from all batches.
    """

    # Initialize data_list as a list of empty lists
    data_list = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            for i, data in enumerate(batch):
                # Check if the data is a tensor. If so, convert it to a numpy array
                if torch.is_tensor(data):
                    data = data.cpu().numpy()
                # If the data is a tuple, convert it to a numpy array
                elif isinstance(data, tuple):
                    data = np.array(data)

                # If it's the first batch, create an empty list for each type of data
                if len(data_list) <= i:
                    data_list.append([])

                data_list[i].append(data)

    # Convert lists to numpy arrays
    for i in range(len(data_list)):
        input=data_list[i]
        #data_list[i] = np.concatenate(data_list[i], axis=0)
        if isinstance(input[0], np.ndarray):  # Case 1
            output = np.concatenate(input, axis=0)
        elif isinstance(input[0], list):  # Case 2
            output = [np.concatenate([input[i][j] for i in range(len(input))], axis=0) for j in range(len(input[0]))]

        data_list[i]=output
    return data_list
