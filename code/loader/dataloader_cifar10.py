import pickle  # For loading serialized data files
from tqdm import tqdm  # For displaying progress bars
import os  # For interacting with the file system
from torch.utils import data  # For dataset utilities in PyTorch
import numpy as np  # For handling arrays and numerical data
import torch  # Main PyTorch library
import torchvision.transforms as transforms  # For image preprocessing and augmentation
from PIL import Image  # For handling image operations

# Custom Dataset Class for CIFAR-10
class DataloaderCifar10(data.Dataset):
    def __init__(self, img_size=32, is_transform=False, split='train'):
        """
        Initialize the DataloaderCifar10 object.
        Args:
            img_size (int): Size to which the images should be resized.
            is_transform (bool): Whether to apply data augmentation or not.
            split (str): The dataset split ('train', 'val', or 'test').
        """
        self.split = split  # Train, validation, or test split
        self.img_size = img_size  # Desired image size
        self.is_transform = is_transform  # Whether to apply transformations or not

        # Transformations for training set: resizing, cropping, flipping, and converting to tensor
        self.transform_train = transforms.Compose([
            transforms.Resize(img_size),  # Resize image to the desired size
            transforms.RandomCrop(img_size, padding=4),  # Randomly crop image with padding
            transforms.RandomHorizontalFlip(),  # Randomly flip image horizontally
            transforms.ToTensor(),  # Convert image to tensor format and normalize pixel values
        ])

        # Transformations for test set: resizing and converting to tensor
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),  # Resize image to the desired size
            transforms.ToTensor(),  # Convert image to tensor format
        ])

        self.data_list = []  # Will store the image data

    # Helper function to load pickled data files
    def unpickle(self, file):
        """
        Unpickle the CIFAR-10 data file.
        Args:
            file (str): The path to the file to unpickle.
        Returns:
            dict: The unpickled data dictionary.
        """
        with open(file, 'rb') as fo:  # Open the file in binary read mode
            dict = pickle.load(fo, encoding='bytes')  # Load the dictionary from the file
        return dict

    # Load the data based on the split (train, val, or test)
    def load_data(self, data_root):
        """
        Load CIFAR-10 data from the specified directory.
        Args:
            data_root (str): The directory where CIFAR-10 data is stored.
        Returns:
            tuple: A tuple of image data and labels.
        """
        all_labels = []  # To store all labels
        all_data = []  # To store all image data

        # Decide which files to load based on the split
        if self.split in ['train', 'val']:
            # CIFAR-10 training data consists of 5 batches
            file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        elif self.split == 'test':
            # CIFAR-10 test data is in a single batch
            file_list = ['test_batch']
        else:
            # Raise an error if an invalid split is specified
            raise ValueError('wrong split! the split should be chosen from train/val/test!')

        # Iterate through each batch file and load data
        for i, file_name in enumerate(file_list):
            cur_batch = self.unpickle(os.path.join(data_root, file_name))  # Unpickle current batch
            data = cur_batch[b'data']  # Extract image data (shape: [10000, 3072] -> 10000 images of size 32x32x3)
            labels = cur_batch[b'labels']  # Extract labels (list of 10000 labels)
            all_data.append(data)  # Append current batch data to the full dataset
            all_labels = all_labels + labels  # Append current batch labels to the full label list

        # Concatenate and reshape the data into proper format (RGB images)
        all_data = np.concatenate(all_data, axis=0)  # Combine all batches into one
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32)  # Reshape into [num_images, 3, 32, 32] (CHW format)
        all_data = all_data.transpose((0, 2, 3, 1))  # Convert from CHW to HWC (common format for image processing)
        all_data = list(all_data)  # Convert to list format

        # Split data into train, validation, and test sets based on the specified split
        if self.split == 'train':
            # Use the first 45,000 images for training
            self.data_list = all_data[0:45000]
            self.label_list = all_labels[0:45000]
        elif self.split == 'val':
            # Use the last 5,000 images for validation
            self.data_list = all_data[45000:]
            self.label_list = all_labels[45000:]
        elif self.split == 'test':
            # Use all images for testing
            self.data_list = all_data
            self.label_list = all_labels

        # Print the number of samples loaded for the current split
        print('[INFO] {} set loaded, {} samples in total.'.format(self.split, len(self.data_list)))
        return self.data_list, self.label_list  # Return the data and labels

    # Required method for PyTorch Dataset class: returns the length of the dataset
    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.data_list)  # Return the number of images in the dataset

    # Required method for PyTorch Dataset class: returns a single item (image and label)
    def __getitem__(self, index):
        """
        Retrieve a single sample (image and label) from the dataset.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        img = self.data_list[index]  # Get the image (in HWC format, RGB, numpy array)
        label = self.label_list[index]  # Get the corresponding label

        # Convert numpy array image to PIL Image format
        img = Image.fromarray(img)

        # Apply data augmentation if required and it's the training set
        if self.is_transform and self.split == 'train':
            img = self.transform_train(img)  # Apply training transformations (CHW format)
        else:
            img = self.transform_test(img)  # Apply test/validation transformations (CHW format)

        # Normalize the image to the range [-1, 1] (commonly used in neural networks)
        img = ((img - 0.5) / 0.5)

        return img, label  # Return the transformed image and label
