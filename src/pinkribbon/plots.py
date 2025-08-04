"""
Module de visualisation d'images médicales et d'analyse spectrale.

Ce module fournit des fonctions spécialisées pour la visualisation d'images
médicales, l'analyse des spectres de couleur, et l'amélioration d'images
par transformation CLAHE. Il est optimisé pour l'analyse de datasets
d'images médicales dans le contexte de détection de cancer.

Fonctions:
    plot_sample: Affichage d'échantillons d'images par classe
    get_specter: Analyse spectrale d'une image individuelle
    get_overlaped_specter: Analyse spectrale superposée de plusieurs images
    get_mean_specter: Affichage des intensités moyennes par canal
    get_overlaped_mean_specter: Intensités moyennes superposées de plusieurs images
    clahe_transform: Transformation CLAHE pour amélioration du contraste
    plot_clahe: Visualisation comparative avant/après CLAHE

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


def plot_sample(dataloader, samples, selected_class, seed=None):
    """
    Affiche un échantillon d'images d'une classe spécifique.
    
    Cette fonction extrait et affiche plusieurs images aléatoires d'une classe
    donnée à partir d'un dataloader. Elle est utile pour l'exploration visuelle
    des données et la vérification de la qualité des échantillons.
    
    Args:
        dataloader: Générateur de données contenant les images
        samples (int): Nombre d'échantillons à afficher
        selected_class: Classe à échantillonner (0 ou 1 pour classification binaire)
        seed (int, optional): Graine pour la reproductibilité. Defaults to None.
    
    Returns:
        dict: Dictionnaire contenant les échantillons sélectionnés avec leurs clés
    
    Example:
        >>> sample_dict = plot_sample(train_generator, 5, class=1, seed=42)
        >>> # Affiche 5 images de la classe 1 avec titre
    """
    np.random.seed(seed)
    sample = dataloader.get_sample(samples, selected_class)
    figure, ax = plt.subplots(1, samples, figsize=(samples, 2))
    i = 0
    for key, img in sample.items():
        img = img[0].array
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_title(f"{key}")
        i += 1
    
    plt.suptitle(f"Echantillon de la classe {selected_class}")
    return sample


def get_specter(img):
    """
    Génère et affiche l'analyse spectrale des canaux de couleur d'une image.
    
    Cette fonction crée un graphique de densité de probabilité (KDE) pour
    chaque canal de couleur (Rouge, Vert, Bleu) afin d'analyser la distribution
    des intensités de pixels dans l'image.
    
    Args:
        img (np.ndarray): Image en format RGB de forme (H, W, 3)
    
    Note:
        - Utilise seaborn.kdeplot pour l'estimation de densité par noyau
        - Fixe les limites Y à [0, 0.25] et X à [0, 255]
        - Colore chaque canal selon sa couleur respective
    
    Example:
        >>> get_specter(medical_image)
        >>> # Affiche le spectre de couleur de l'image médicale
    """
    red_channel = img[:, :, 0].reshape(-1)
    green_channel = img[:, :, 1].reshape(-1)
    blue_channel = img[:, :, 2].reshape(-1)
    
    fig = plt.figure(figsize=(8, 3))
    sns.kdeplot(red_channel, color="red", fill=True)
    sns.kdeplot(green_channel, color="green", fill=True)
    g = sns.kdeplot(blue_channel, color="blue", fill=True)
    g.set_ylim(0, 0.25)
    g.set_xlim(0, 255)
    g.set_ylabel("Intensité")


def get_overlaped_specter(sample):
    """
    Génère une analyse spectrale superposée pour plusieurs images.
    
    Cette fonction crée un graphique unique montrant les spectres de couleur
    de toutes les images d'un échantillon superposés, permettant de comparer
    facilement les distributions spectrales entre différentes images.
    
    Args:
        sample (dict): Dictionnaire d'images où chaque valeur contient
                      un objet avec un attribut 'array' (format: img[0].array)
    
    Note:
        - Toutes les courbes sont affichées sur le même graphique
        - Utile pour comparer les caractéristiques spectrales entre images
        - Les couleurs se superposent selon leur transparence
    
    Example:
        >>> sample_dict = dataloader.get_sample(5, class=0)
        >>> get_overlaped_specter(sample_dict)
        >>> # Affiche les spectres superposés de 5 images
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    for img in sample.values():
        img = img[0].array
        red_channel = img[:, :, 0].reshape(-1)
        green_channel = img[:, :, 1].reshape(-1)
        blue_channel = img[:, :, 2].reshape(-1)
    
        sns.kdeplot(red_channel, color="red", fill=True)
        sns.kdeplot(green_channel, color="green", fill=True)
        g = sns.kdeplot(blue_channel, color="blue", fill=True)
    g.set_ylim(0, 0.25)
    g.set_xlim(0, 255)
    g.set_ylabel("Intensité")


def get_mean_specter(img, ax=None):
    """
    Affiche les intensités moyennes de chaque canal de couleur sous forme de lignes verticales.
    
    Cette fonction calcule la valeur moyenne de chaque canal de couleur
    et l'affiche comme une ligne verticale sur un graphique, fournissant
    une représentation simple de la tonalité générale de l'image.
    
    Args:
        img (np.ndarray): Image en format RGB de forme (H, W, 3)
        ax (matplotlib.axes.Axes, optional): Axes matplotlib pour le tracé.
                                           Si None, utilise les axes courants.
    
    Note:
        - Rouge: ligne verticale rouge à la position de l'intensité moyenne rouge
        - Vert: ligne verticale verte à la position de l'intensité moyenne verte
        - Bleu: ligne verticale bleue à la position de l'intensité moyenne bleue
        - Échelle X fixée à [0, 255]
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> get_mean_specter(image, ax)
        >>> plt.show()
    """
    red_channel = img[:, :, 0].mean()
    green_channel = img[:, :, 1].mean()
    blue_channel = img[:, :, 2].mean()
    ax.axvline(red_channel, 0, 1, c="red") 
    ax.axvline(green_channel, 0, 1, c="green")
    ax.axvline(blue_channel, 0, 1, c="blue")
    ax.set_xlim(0, 255)


def get_overlaped_mean_specter(sample): 
    """
    Affiche les intensités moyennes superposées pour plusieurs images.
    
    Cette fonction crée un graphique montrant les intensités moyennes
    de tous les canaux de couleur pour toutes les images d'un échantillon,
    permettant de comparer rapidement les tonalités générales.
    
    Args:
        sample (dict): Dictionnaire d'images où chaque valeur contient
                      un objet avec un attribut 'array' (format: img[0].array)
    
    Note:
        - Chaque image contribue trois lignes verticales (R, G, B)
        - Utile pour identifier des patterns dans les tonalités d'un groupe d'images
        - Les lignes se superposent selon leur position sur l'échelle des intensités
    
    Example:
        >>> sample_dict = dataloader.get_sample(10, class=1)
        >>> get_overlaped_mean_specter(sample_dict)
        >>> # Affiche les intensités moyennes de 10 images superposées
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.set_xlim(0, 255)
    for img in sample.values():
        img = img[0].array
        red_channel = img[:, :, 0].mean()
        green_channel = img[:, :, 1].mean()
        blue_channel = img[:, :, 2].mean()
        ax.axvline(red_channel, 0, 1, c="red") 
        ax.axvline(green_channel, 0, 1, c="green")
        ax.axvline(blue_channel, 0, 1, c="blue") 
        

def clahe_transform(img):
    """
    Applique la transformation CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Cette fonction améliore le contraste local d'une image en appliquant CLAHE
    sur le canal de luminance dans l'espace colorimétrique LAB, préservant
    ainsi les informations chromatiques tout en améliorant la visibilité.
    
    Args:
        img (np.ndarray): Image d'entrée en format BGR (OpenCV standard)
    
    Returns:
        np.ndarray: Image avec contraste amélioré en format RGB
    
    Paramètres CLAHE:
        - clipLimit: 2.0 (limite le contraste pour éviter la sur-amplification)
        - tileGridSize: (8, 8) (taille des tuiles pour l'adaptation locale)
    
    Note:
        - Conversion BGR→LAB→BGR→RGB pour compatibilité avec matplotlib
        - Préserve les informations de couleur en ne modifiant que la luminance
    
    Example:
        >>> enhanced_img = clahe_transform(original_bgr_image)
        >>> # Returns RGB image with enhanced contrast
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)


def plot_clahe(img_rgb, enhanced_img_rgb):
    """
    Affiche une comparaison côte à côte avant/après transformation CLAHE.
    
    Cette fonction crée une visualisation comparative montrant l'image
    originale et sa version améliorée par CLAHE sur un seul graphique,
    facilitant l'évaluation de l'amélioration du contraste.
    
    Args:
        img_rgb (np.ndarray): Image originale en format RGB
        enhanced_img_rgb (np.ndarray): Image améliorée par CLAHE en format RGB
    
    Layout:
        - Subplot gauche: Image originale (titre: "Original")
        - Subplot droit: Image améliorée (titre: "Amélioré")
        - Axes désactivés pour une visualisation claire
        - Taille de figure: (4, 2)
    
    Example:
        >>> original = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        >>> enhanced = clahe_transform(bgr_image)
        >>> plot_clahe(original, enhanced)
        >>> # Affiche la comparaison avant/après
    """
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Amélioré")
    plt.imshow(enhanced_img_rgb)
    plt.axis('off')
    plt.show()


__all__ = [
    "plot_sample",
    "get_specter", 
    "get_overlaped_specter",
    "get_mean_specter",
    "get_overlaped_mean_specter",
    "clahe_transform",
    "plot_clahe"
]