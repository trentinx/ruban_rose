from .images import MedImage, clahe, random_transformations
from keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    _batch_size = 4
    dim = (50,50)
    n_channels = 3
    
    def __init__(self, zip_path):
        super().__init__()
        self.zip_path = zip_path 
        self.indexes = []
        self.__load_data()
    
    def __len__(self):
        return 4 * self.half_len // self.batch_size

    def __getitem__(self, index):
        indexes = []
        start = index * self._batch_size
        end =  (index + 1 ) * self._batch_size
        for ids in self.indexes:
           indexes += (ids[start:end]).tolist()
        np.random.shuffle(indexes)
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        #Re-init indexes list
        for indexes_list in self.indexes:   
            np.random.shuffle(indexes_list)

    def __load_data(self):
        self.data = []
        MedImage.open_zfile(self.zip_path)
        self.data = np.array([MedImage(f) for  f in MedImage.zfile.namelist() if not f.startswith("IDC_regular_ps50_idx5") and f.lower().endswith(('.png'))])
        self.indexes.append(np.where([obj.label == 0 for obj in self.data])[0])
        self.indexes.append(np.where([obj.label == 1 for obj in self.data])[0])
        self.half_len = max([len(l) for l in self.indexes])


    def __find_minority_class(self):
        minority = None
        ratio = len(self.indexes[0])/len(self.indexes[1])
        if ratio < 1:
            minority = 0
        elif ratio > 1:
            minority = 1
        else:
            print("Classes are already balanced")
        return minority

    def balance(self):
        minority = self.__find_minority_class() 
        if minority is not None:
            balanced_indexes = []
            for i in range(2):
                if minority == i:
                    balanced_indexes.append([])
                    length = len(self.indexes[i])
                    while length < self.half_len:
                        balanced_indexes[i] += self.indexes[i][:min(length, self.half_len - length)].tolist()
                        length = len(balanced_indexes[i])
                    balanced_indexes[i] = np.array(balanced_indexes[i])
                else:
                    balanced_indexes.append(self.indexes[i])
            self.indexes = balanced_indexes
                   
    @property 
    def batch_size(self):
        return 2 * self._batch_size
         
    @batch_size.setter
    def batch_size(self, value):
        if value % 2 > 0:
            print(f"batch_size parameter must be odd, keeping curren value {self.batch_size}.")
        else:
            self._batch_size = value // 2

    def sample(self, size, reverse=False, shuffle=False):
        indexes = []
        if isinstance(size, int):
            output_size = min(size // 2, self.half_len)
        elif isinstance(size, float):
            output_size = int(size*self.half_len)
        else:
            return None
        for labels in self.indexes:
            indexes.append(self.__sample_label(labels,output_size, reverse, shuffle))
        output = DataGenerator(self.zip_path)
        output.indexes = indexes
        output.half_len = output_size
        return output

    def __sample_label(self, labels, size, reverse, shuffle):
        output = labels
        if shuffle:
            np.random.shuffle(output)
        if reverse:
            output = output[::-1]
        return output[:min(size,len(output))]

    def load_img(self, img_id):
        data = self.data[img_id]
        y = data.label
        X = data.resized(self.dim)
        return X,y
        
    def __data_generation(self, indexes):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.uint8)
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, img_id in enumerate(indexes):

            # Get image in array format and label
            img, label = self.load_img(img_id)
            
            # Transform / augment data
            img = random_transformations(img)
            img = clahe(img)
            
            # Store sample
            X[i,], y[i] = img, label

        # Normalizarion
        X = X / 255.
            
        return X, y

            
__all__ = ["DataGenerator"]   
        
        