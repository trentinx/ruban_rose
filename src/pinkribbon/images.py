from keras import Sequential, layers
from keras.ops import convert_to_tensor

import cv2
import numpy as np
import os
import zipfile

class MedImage:
    zfile = None
    
    def __init__(self, image_path):
        self.image_path = image_path

    @classmethod
    def open_zfile(cls, zpath):
        MedImage.zfile = zipfile.ZipFile(zpath, 'r')

    
    @property
    def array(self):
        # Charger l'image directement depuis le zip
        with self.zfile.open(self.image_path) as img_file:
            img_data = img_file.read()
            img_array = np.frombuffer(img_data, np.uint8)
            img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_array  = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
            return img_array

    @property
    def label(self):
        return int(os.path.dirname(self.image_path).split('/')[-1])

    def resized(self, dim):
        return resize(self.array, dim)

def clahe(array):
    lab = cv2.cvtColor(array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

def random_transformations(array):
    tensor = convert_to_tensor(array)
    output = Sequential([
    layers.RandomRotation(0.1, dtype=np.uint8),
    layers.RandomTranslation(0.2, 0.2,dtype=np.uint8),
    layers.RandomZoom(0.2, 0.2,dtype=np.uint8),
    layers.RandomFlip(dtype=np.uint8),
    ])(tensor)
    return np.array(output)

def resize(array, dim):
    #cv2.INTER_NEAREST : rapide, faible qualité
    #cv2.INTER_LINEAR : bon pour agrandir (par défaut)
    #cv2.INTER_AREA : bon pour réduire (recommandé pour downscaling)
    #cv2.INTER_CUBIC, cv2.INTER_LANCZOS4 : plus lent, meilleure qualité
    
    DOWNSCALE = {False:  cv2.INTER_CUBIC, True: cv2.INTER_AREA}
    src_dim = array.shape[:2]
    dbool = (src_dim[0] <=  dim[0])  and (src_dim[1] <=  dim[1])
    return cv2.resize(array, dim, interpolation = DOWNSCALE[dbool])

__all__ = ["MedImage", "clahe", "random_transformations"]