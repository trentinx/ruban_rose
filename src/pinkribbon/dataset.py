from PIL import Image, ImageOps, ImageEnhance
from keras.applications.resnet50 import preprocess_input
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
import io
import numpy as np
import random
import zipfile


class DataGenerator(Sequence):
    zipfile = None

    
    def __init__(self, zip_path=None, transform=None, max_samples_per_class=None, seed=None, experimentation="resnet50_1"):
        super().__init__()
        self.transform = transform
        self.samples = []
        if zip_path is not None:
            self.open_zfile(zip_path)
        self.shuffle = False    
        self.batch_size = 8
        self.transform = False
        self.experimentation = experimentation
        
        self.load_img = {"resnet50_1": self.load_img_1,
                         "resnet50_2": self.load_img_2,
                         "resnet50_3": self.load_img_3,
                         "resnet50_5": self.load_img_2
        }
                         
        
        file_list = np.array([f for f in DataGenerator.zfile.namelist() if not f.startswith("IDC_regular_ps50_idx5") and f.lower().endswith('.png')], dtype=object)
    
        labels =  np.array([int(f.split('/')[-2]) for f in file_list ],dtype=int)

        for i in range(2):
            indexes = np.where(labels==i)[0].tolist()
            samples = [(f, i) for f in file_list[indexes]]
            if max_samples_per_class is not None:
                samples = samples[:max_samples_per_class]
            self.samples += samples
    
        if seed is not None:
            np.random.seed(seed=42)
        np.random.shuffle(self.samples)
        
    @classmethod
    def open_zfile(cls, zpath):
        DataGenerator.zfile = zipfile.ZipFile(zpath, 'r')

    def __len__(self):
        return len(self.samples) // self.batch_size

    def load_img_1(self, path):
        image_bytes = self.zfile.read(path)
        l,a,b = Image.open(io.BytesIO(image_bytes))\
                   .resize((50,50),resample = Image.BICUBIC)\
                   .convert("LAB")\
                   .split()

        l = ImageOps.equalize(l)
        img = Image.merge('LAB', (l, a, b))\
                   .convert("RGB")

        return np.array(img)/255.
    
    def load_img_2(self, path):
        image_bytes = self.zfile.read(path)
        img = Image.open(io.BytesIO(image_bytes))\
                   .resize((224,224),resample = Image.BICUBIC)\
                   .convert("RGB")
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.squeeze(img, axis=0)
        return img
    
    def load_img_3(self, path):
        image_bytes = self.zfile.read(path)
        l,a,b = Image.open(io.BytesIO(image_bytes))\
                   .resize((224,224),resample = Image.BICUBIC)\
                   .convert("LAB") \
                   .split()

        l = ImageOps.equalize(l)
        img = Image.merge('LAB', (l, a, b))\
                   .convert("RGB")
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.squeeze(img, axis=0)
        return img
        
        
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end =  (idx + 1 ) * self.batch_size
        samples = self.samples[start:end]
        X = np.stack([self.load_img[self.experimentation](s[0]) for s in samples]).astype(np.float16)
        y = np.array([s[1] for s in samples], dtype=np.uint8)
        return X,y
    
    def on_epoch_end(self):
        #Re-init indexes list
        if self.shuffle:   
            np.random.shuffle(self.samples)
    

    def train_test_split(self, test_size=0.2):
        train_samples, val_samples = train_test_split(
            self.samples,
            test_size=test_size,
            stratify=[label for _, label in self.samples],  # garder le ratio des classes
            random_state=42
        )
        train_generator = DataGenerator()
        train_generator.samples = train_samples
        
        val_generator =  DataGenerator()
        val_generator.samples = val_samples
        return train_generator, val_generator
            
__all__ = ["DataGenerator"]   
        
        