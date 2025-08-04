"""
Module de traitement d'images médicales pour l'analyse de cancer.

Ce module fournit des classes et fonctions pour le chargement, la manipulation
et la transformation d'images médicales à partir d'archives ZIP. Il inclut
des fonctionnalités d'amélioration d'images (CLAHE), de transformations
aléatoires pour l'augmentation de données, et de redimensionnement intelligent.

Classes:
    MedImage: Classe pour représenter et manipuler une image médicale

Fonctions:
    clahe: Amélioration du contraste adaptatif par histogramme limité
    random_transformations: Transformations aléatoires pour l'augmentation de données
    resize: Redimensionnement intelligent d'images
"""

from keras import Sequential, layers
from keras.ops import convert_to_tensor

import cv2
import numpy as np
import os
import zipfile

class MedImage:
    """
    Classe pour représenter et manipuler une image médicale.
    
    Cette classe permet de charger des images directement depuis des archives ZIP,
    d'extraire automatiquement les labels depuis la structure de dossiers,
    et de fournir diverses méthodes pour l'analyse et la transformation d'images.
    
    Attributes:
        zfile (zipfile.ZipFile): Archive ZIP partagée entre toutes les instances
        image_path (str): Chemin de l'image dans l'archive ZIP
    
    Example:
        >>> MedImage.open_zfile("data/images.zip")
        >>> img = MedImage("cancer/image_001.jpg")
        >>> array = img.array
        >>> label = img.label
    """
    
    zfile = None
    
    def __init__(self, image_path):
        """
        Initialise une instance MedImage.
        
        Args:
            image_path (str): Chemin de l'image dans l'archive ZIP,
                            incluant la structure de dossiers pour l'extraction du label.
        """
        self.image_path = image_path

    @classmethod
    def open_zfile(cls, zpath):
        """
        Ouvre un fichier ZIP pour accès partagé par toutes les instances.
        
        Cette méthode de classe doit être appelée avant de créer des instances
        MedImage pour définir l'archive ZIP source.
        
        Args:
            zpath (str): Chemin vers le fichier ZIP contenant les images.
        """
        MedImage.zfile = zipfile.ZipFile(zpath, 'r')

    @property
    def array(self):
        """
        Charge et retourne l'image sous forme de tableau numpy.
        
        L'image est chargée directement depuis l'archive ZIP, décodée,
        et convertie du format BGR vers RGB pour compatibilité avec matplotlib.
        
        Returns:
            np.ndarray: Tableau numpy de l'image en format RGB (H, W, 3).
        
        Raises:
            AttributeError: Si aucune archive ZIP n'a été ouverte avec open_zfile().
            KeyError: Si le chemin de l'image n'existe pas dans l'archive.
        """
        # Charger l'image directement depuis le zip
        with self.zfile.open(self.image_path) as img_file:
            img_data = img_file.read()
            img_array = np.frombuffer(img_data, np.uint8)
            img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            return img_array

    @property
    def label(self):
        """
        Extrait le label de classification depuis le nom du dossier parent.
        
        Le label est extrait automatiquement depuis la structure de dossiers
        de l'image dans l'archive ZIP (ex: "0/image.jpg" -> label 0.0).
        
        Returns:
            float: Label de classification (0.0 ou 1.0 pour classification binaire).
        """
        return float(os.path.dirname(self.image_path).split('/')[-1])

    def resized(self, dim):
        """
        Retourne une version redimensionnée de l'image.
        
        Args:
            dim (tuple): Dimensions cibles (largeur, hauteur).
        
        Returns:
            np.ndarray: Image redimensionnée.
        """
        return resize(self.array, dim)
    
    def histogram(self, layer):
        """
        Calcule l'histogramme d'un canal de couleur spécifique.
        
        Args:
            layer (str): Canal de couleur à analyser ('r', 'g', ou 'b').
        
        Returns:
            np.ndarray: Histogramme du canal spécifié (256 bins).
        
        Note:
            Les indices des canaux semblent incorrects dans l'implémentation actuelle.
            Devrait utiliser [:,:,0], [:,:,1], [:,:,2] pour R, G, B respectivement.
        """
        array = self.array
        if layer.lower() == "r":
            return np.bincount(array[:,:,0].reshape(-1), minlength=256)
        if layer.lower() == "g":
            return np.bincount(array[:,:,1].reshape(-1), minlength=256)
        if layer.lower() == "b":
            return np.bincount(array[:,:,2].reshape(-1), minlength=256)

def clahe(array):
    """
    Applique l'amélioration du contraste adaptatif par histogramme limité (CLAHE).
    
    Cette fonction améliore le contraste local de l'image en appliquant CLAHE
    sur le canal de luminance dans l'espace colorimétrique LAB, préservant
    ainsi les informations de couleur.
    
    Args:
        array (np.ndarray): Image d'entrée en format RGB (H, W, 3).
    
    Returns:
        np.ndarray: Image avec contraste amélioré en format RGB.
    
    Note:
        - Limite de clip: 2.0 (évite la sur-amplification du bruit)
        - Taille de grille: 8x8 (bon compromis entre efficacité et qualité)
    
    Example:
        >>> img_enhanced = clahe(original_image)
    """
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe_obj.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

def random_transformations(array):
    """
    Applique des transformations aléatoires pour l'augmentation de données.
    
    Cette fonction applique une séquence de transformations géométriques
    aléatoires pour augmenter la diversité du dataset d'entraînement.
    
    Args:
        array (np.ndarray): Image d'entrée en format RGB (H, W, 3).
    
    Returns:
        np.ndarray: Image transformée avec les mêmes dimensions.
    
    Transformations appliquées:
        - Rotation aléatoire: ±10% (0.1 radians)
        - Translation aléatoire: ±20% en X et Y
        - Zoom aléatoire: ±20%
        - Retournement horizontal/vertical aléatoire
    
    Example:
        >>> augmented_img = random_transformations(original_image)
    """
    tensor = convert_to_tensor(array)
    output = Sequential([
        layers.RandomRotation(0.1, dtype=np.uint8),
        layers.RandomTranslation(0.2, 0.2, dtype=np.uint8),
        layers.RandomZoom(0.2, 0.2, dtype=np.uint8),
        layers.RandomFlip(dtype=np.uint8),
    ])(tensor)
    return np.array(output)

def resize(array, dim):
    """
    Redimensionne une image avec interpolation adaptative.
    
    Cette fonction choisit automatiquement la méthode d'interpolation
    optimale selon que l'image est agrandie ou réduite, garantissant
    la meilleure qualité possible.
    
    Args:
        array (np.ndarray): Image d'entrée (H, W, C).
        dim (tuple): Dimensions cibles (largeur, hauteur).
    
    Returns:
        np.ndarray: Image redimensionnée aux dimensions spécifiées.
    
    Méthodes d'interpolation:
        - INTER_CUBIC: Pour l'agrandissement (meilleure qualité)
        - INTER_AREA: Pour la réduction (évite l'aliasing)
    
    Example:
        >>> resized_img = resize(image, (224, 224))
    """
    # cv2.INTER_NEAREST : rapide, faible qualité
    # cv2.INTER_LINEAR : bon pour agrandir (par défaut)
    # cv2.INTER_AREA : bon pour réduire (recommandé pour downscaling)
    # cv2.INTER_CUBIC, cv2.INTER_LANCZOS4 : plus lent, meilleure qualité
    
    DOWNSCALE = {False: cv2.INTER_CUBIC, True: cv2.INTER_AREA}
    src_dim = array.shape[:2]
    is_downscale = (src_dim[0] >= dim[1]) and (src_dim[1] >= dim[0])
    return cv2.resize(array, dim, interpolation=DOWNSCALE[is_downscale])

__all__ = ["MedImage", "clahe", "random_transformations", "resize"]