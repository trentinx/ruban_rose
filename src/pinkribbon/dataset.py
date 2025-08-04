from .images import MedImage
from PIL import Image, ImageOps
from keras.applications.resnet50 import preprocess_input
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
import io
import numpy as np
import zipfile
import pandas as pd


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
    
    
class ZipImageDataLoader():
    def __init__(self, zip_path):
        self.zfile = zipfile.ZipFile(zip_path, 'r')

        # Liste des fichiers image avec leurs labels
        self.image_files = [f for f in self.zfile.namelist() if not f.startswith("IDC_regular_ps50_idx5") and f.lower().endswith(('.png'))]

        # Extraire les labels depuis les noms de dossier
        self.classes = np.array([int(f.split('/')[-2]) for f in self.image_files])
    
    def __len__(self):
        return len(self.image_files)
    
    def get_sample(self, sample_size, selected_class):
        indexes = np.where(self.classes == selected_class)[0]
        np.random.shuffle(indexes)
        indexes = indexes[:sample_size]
        images = [self.__getitem__(idx) for idx in indexes]
        return dict(zip(indexes,images))
    
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_name = image_path.split('/')[-2]
        label = int(label_name)
        return MedImage(image_path), label 
    
def build_pixels_dataframe(dataloader, max_nb_per_classe=1000):
    columns = ["classe"] + [f"{c}_{i}" for c in ["r","g","b" ]for i in range(256)]
    samples = []
    for i in range(2):
        for k,v  in dataloader.get_sample(max_nb_per_classe,i).items():
            hist = np.concatenate((
                np.atleast_1d(v[1]),
                v[0].histogram("r"),
                v[0].histogram("g"),
                v[0].histogram("b")))
            samples.append(hist)
    return pd.DataFrame(np.array(samples),columns=columns)
            
__all__ = ["DataGenerator","ZipImageDataLoader","build_pixels_dataframe"]   
        
        