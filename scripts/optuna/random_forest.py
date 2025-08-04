"""
Script d'optimisation d'hyperparamètres pour modèle ResNet50 avec Optuna.

Ce script implémente un pipeline d'optimisation multi-objectifs pour un modèle
de classification binaire basé sur ResNet50 pré-entraîné. Il utilise Optuna
pour optimiser simultanément le recall (à maximiser) et la loss (à minimiser).

Le modèle utilise ResNet50 comme extracteur de caractéristiques gelé, suivi
de couches denses personnalisées pour la classification binaire avec sigmoid.


Prétraitement appliqué:
Images en 50x50x3 en RGB + CLAHE + normalisation.

Usage:
    python resnet50_1.py

Dependencies:
    - keras (TensorFlow)
    - optuna
    - pinkribbon (module local)
    - numpy, pandas (via dependencies)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from pinkribbon.callbacks import *
from pinkribbon.dataset import build_pixels_dataframe, ZipImageDataLoader
from pinkribbon.images import MedImage
import gc
import optuna

NB_TRIALS = 20
STUDY_NAME = "random_forest" 


def launch_study(nb_trials, study_name):
    """
    Lance une étude d'optimisation Optuna pour l'hyperparameter tuning.
    
    Cette fonction initialise les générateurs de données, configure les paramètres
    d'entraînement et de validation, puis lance l'optimisation Optuna sur le
    nombre spécifié d'essais.
    
    Args:
        nb_trials (int): Nombre d'essais à effectuer pour l'optimisation.
        study_name (str): Nom de l'étude utilisé pour l'expérimentation
                         et passé au DataGenerator.
    
    Note:
        - Utilise un split train/test de 80/20
        - Maximum de 1500 échantillons par classe
        - Seed fixé à 42 pour la reproductibilité
        - Le générateur de validation a une taille de batch de 1
        - Le générateur d'entraînement est mélangé
    """
    print("Initializing data generators ", end="")
    dataloader = ZipImageDataLoader("data/BHI.zip")
    MedImage.open_zfile("data/BHI.zip")
    df = build_pixels_dataframe(dataloader,1500)
    X = df.iloc[:,1:]
    y  = df.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42, stratify=y)
    
    study.optimize(lambda trial : 
                        objective(
                            trial, 
                            X_train,
                            X_test,
                            y_train,
                            y_test
                        ),
                        gc_after_trial=True, 
                        n_trials=nb_trials)
    

def objective(trial, X_train, X_test, y_train, y_test):
    """
    Fonction objectif pour l'optimisation Optuna multi-objectifs.
    
    Cette fonction définit et entraîne un modèle avec des hyperparamètres
    suggérés par Optuna, puis retourne les métriques à optimiser.
    L'optimisation vise à maximiser le recall et minimiser la loss.
    
    Args:
        trial (optuna.Trial): Objet trial Optuna pour suggérer les hyperparamètres.
        train_generator (DataGenerator): Générateur de données d'entraînement.
        val_generator (DataGenerator): Générateur de données de validation.
    
    Returns:
        tuple: (recall, loss) - Tuple contenant le recall de validation (à maximiser)
               et la loss de validation (à minimiser).
    
    Hyperparamètres optimisés:
        - lr (float): Taux d'apprentissage (1e-5 à 1e-2, échelle log)
        - batch_size (int): Taille du batch (16, 32, ou 64)
        - optimizer (str): Type d'optimiseur ("Adam" ou "SGD")
        - dropout_rate (float): Taux de dropout (0.2 à 0.5)
        - weight_decay (float): Décroissance des poids (1e-6 à 1e-3, échelle log)
        - momentum (float): Momentum pour SGD seulement (0.8 à 0.99)
    
    Raises:
        optuna.exceptions.TrialPruned: Si l'essai est élagué par le callback de pruning.
    """
    
    
    n_estimators = trial.suggest_int('n_estimators',50, 1000)  # nombre d'arbres
    max_depth = trial.suggest_int('max_depth',10,100)        # profondeur max des arbres
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)        # nombre min d'échantillons pour splitter un noeud
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)           # nombre min d'échantillons dans une feuille
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    gc.collect()
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features)
    
    try:
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       score = recall_score(y_test,y_pred)
       
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()

    return score

if __name__ == "__main__":   
    # Création d'un pruner MedianPruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=3,  
        interval_steps=1
    )
    
    storage = optuna.storages.RDBStorage(
    url='sqlite:///results/optunarf.db',
    heartbeat_interval=60,
    grace_period=120,
    failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )
    
    
    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name=STUDY_NAME,
        load_if_exists=True,
        pruner=pruner
    )
    

    
    print(f"Démarrage de l'optimisation avec l'étude: {STUDY_NAME}")
    print(f"Nombre d'essais prévus: {NB_TRIALS}")
    
    launch_study(NB_TRIALS, STUDY_NAME)
    
    print("Number of finished trials: {}".format(len(study.trials)))