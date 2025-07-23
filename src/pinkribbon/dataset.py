
from PIL import Image
from io import BytesIO
from keras.utils import Sequence

import numpy as np
import os
import zipfile

class DataGenerator(Sequence):
    """
    A Keras data generator for loading breast cancer histopathology images from a zip file.
    
    This class extends Keras Sequence to provide efficient batch loading of IDC (Invasive Ductal Carcinoma)
    images for binary classification. It supports balanced sampling from both classes (0: non-IDC, 1: IDC)
    and handles data loading directly from a zip archive.
    
    Attributes:
        zip_path (str): Path to the zip file containing the dataset
        rate (float): Fraction of the dataset to use (default: 0.1)
        batch_size (int): Number of samples per batch (default: 4)
        shuffle (bool): Whether to shuffle data after each epoch (default: True)
        n_classes (int): Number of classes (fixed to 2 for binary classification)
        dim (tuple): Image dimensions (50, 50)
        n_channels (int): Number of color channels (3 for RGB)
        transform: Optional image transformation function
    """
    def __init__(self, zip_path, rate=0.1, batch_size=4, shuffle=True):
        """
        Initialize the DataGenerator.
        
        Args:
            zip_path (str): Path to the zip file containing IDC images
            rate (float, optional): Fraction of dataset to use. Defaults to 0.1
            batch_size (int, optional): Number of samples per batch. Defaults to 4
            shuffle (bool, optional): Whether to shuffle data after each epoch. Defaults to True
        
        Note:
            The zip file should contain images with paths starting with "IDC_regular_ps50_idx5"
            and organized in directories named "0" (non-IDC) and "1" (IDC) for labeling.
        """
        self.zip_path = zip_path
        self.zfile = zipfile.ZipFile(zip_path, 'r')        
        
        # Liste des noms de fichiers des images
        self.img_ids = np.array([f for f in self.zfile.namelist() if f.startswith("IDC_regular_ps50_idx5") and f.lower().endswith(('.png'))])

        # Liste des labels (même index que la liste des fichiers
        self.labels = np.array([int(os.path.dirname(f).split('/')[-1]) for f in self.img_ids])
        
        # Liste des indexes par classe
        self.label_ids = []
        for i in range(2):
            self.label_ids.append(np.where(self.labels==i)[0])
        
        self.rate = rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = 2
        self.dim = (50,50)
        self.n_channels = 3
        self.transform = False
   
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Update indexes after each epoch.
        
        Resets and optionally shuffles the indexes for each class to ensure
        different sampling order in the next epoch if shuffle is enabled.
        """
        #Réinitialisation de la liste d'indexes
        self.indexes = [] 
        for i in range(self.n_classes):
            self.indexes.append(self.label_ids[i])
            if self.shuffle == True:
                np.random.shuffle(self.indexes[i]) 

    def __len__(self):
        """
        Denote the number of batches per epoch.
        
        Returns:
            int: Number of batches per epoch, calculated to ensure balanced sampling
                 and even number of batches for proper class distribution
        """
        return 2*(int(np.floor(len(self.img_ids) * self.rate / self.batch_size))//2)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        
        Args:
            index (int): Batch index
            
        Returns:
            tuple: A tuple containing:
                - X (numpy.ndarray): Batch of images with shape (batch_size, 50, 50, 3)
                - y (numpy.ndarray): Batch of labels with shape (batch_size,)
                
        Note:
            Ensures balanced sampling by taking equal numbers of samples from each class.
            Handles index overflow by reshuffling when necessary.
        """
        # Generate indexes of the batch
        indexes = []
        batch_size = self.batch_size//2
        for i in range(self.n_classes):
            try:
                indexes += (self.indexes[i][index*batch_size:(index+1)*batch_size]).tolist()
            except IndexError:
                self.indexes[i] = self.label_ids[i]
                if self.shuffle == True:
                    np.random.shuffle(self.indexes[i])
                indexes += (self.indexes[i][index*batch_size:(index+1)*batch_size]).tolist()

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def __data_generation(self, indexes):
        """
        Generate data containing batch_size samples.
        
        Args:
            indexes (list): List of sample indexes to load
            
        Returns:
            tuple: A tuple containing:
                - X (numpy.ndarray): Batch of images with shape (batch_size, 50, 50, 3)
                - y (numpy.ndarray): Batch of labels with shape (batch_size,)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, img_id in enumerate(indexes):
            # Store sample
            X[i,], y[i] = self.load_img(img_id, as_array=True)
        return X, y
            

    def load_img(self, img_id, as_array=False):
        """
        Load an image and its label from the zip file.
        
        Args:
            img_id (int): Index of the image to load
            as_array (bool, optional): Whether to return image as numpy array. Defaults to False
            
        Returns:
            tuple: A tuple containing:
                - image (PIL.Image or numpy.ndarray): The loaded image
                - label_name (int): The class label (0 or 1)
                
        Note:
            Images are converted to RGB format and optionally transformed if transform is set.
        """
        image_path = self.img_ids[img_id]
        label_name = self.labels[img_id]
        # Charger l'image directement depuis le zip
        with self.zfile.open(image_path) as img_file:
            image = Image.open(BytesIO(img_file.read())).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        if as_array:
            image = np.array(image)        
        return image, label_name
    
    