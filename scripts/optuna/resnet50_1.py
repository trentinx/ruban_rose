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

from keras.applications import ResNet50
from keras.backend import clear_session
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from pinkribbon.callbacks import *
from pinkribbon.dataset import DataGenerator
import gc
import optuna

NB_TRIALS = 20
STTUDY_NAME = "resnet50_1"

def build_model(dropout_rate):
    """
    Construit un modèle de classification binaire basé sur ResNet50.
    
    Le modèle utilise ResNet50 pré-entraîné comme extracteur de caractéristiques
    (couches gelées) suivi de couches denses personnalisées avec dropout et
    normalisation par batch pour la classification binaire.
    
    Args:
        dropout_rate (float): Taux de dropout à appliquer dans les couches.
                             Le premier dropout utilise ce taux, le second
                             utilise dropout_rate/2.
    
    Returns:
        keras.Model: Modèle compilé prêt pour l'entraînement avec:
                    - ResNet50 comme base (gelé)
                    - Couches de classification personnalisées
                    - Activation sigmoid pour classification binaire
    """
    resnet = ResNet50( include_top=False,
                       weights="imagenet",
                       input_shape=(50,50,3),
                       pooling = "avg",
                       name="resnet50")
    resnet.trainable = False
    
    model = Sequential()
    model.add(resnet)
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
    model.add(Dense(1, activation='sigmoid'))
    return model


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
    generator = DataGenerator("data/BHI.zip", max_samples_per_class=1500, seed=42, experimentation=study_name)
    train_generator, val_generator = generator.train_test_split(0.2)
    del generator
    val_generator.batch_size = 1
    train_generator.shuffle = True
    
    print(f"Train data size : {len(train_generator) * train_generator.batch_size}")
    print(f"Test data size : {len(val_generator) * val_generator.batch_size}")
    
    study.optimize(lambda trial : 
                        objective(
                            trial, 
                            train_generator = train_generator,
                            val_generator = val_generator,
                        ),
                        gc_after_trial=True, 
                        n_trials=nb_trials)
    

def objective(trial, train_generator, val_generator):
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
    
    EPOCHS = 15  # Augmenté pour permettre plus d'apprentissage
    
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        optimizer = SGD(learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = Adam(learning_rate=lr, weight_decay=weight_decay)
        
    train_generator.batch_size = batch_size
    val_generator.batch_size = batch_size

    clear_session()
    gc.collect()
    model = build_model(dropout_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["recall"])
    
    es = OptunaEarlyStopping()
             
    rp = OptunaReduceLROnPlateau()
                 
    pc = OptunaPruningCallback(trial, monitor='val_recall', mode='max')
    
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[es, rp, pc]
        )
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("history", history.history)
    scores = model.evaluate(val_generator, verbose=0)
    return scores[1], scores[0]

if __name__ == "__main__":   
    # Création d'un pruner MedianPruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=3,  
        interval_steps=1
    )
    
    storage = optuna.storages.RDBStorage(
    url='sqlite:///results/optuna.db',
    heartbeat_interval=60,
    grace_period=120,
    failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )
    
    
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        storage=storage,
        study_name=STTUDY_NAME,
        load_if_exists=True,
        pruner=pruner
    )
    

    
    print(f"Démarrage de l'optimisation avec l'étude: {STTUDY_NAME}")
    print(f"Nombre d'essais prévus: {NB_TRIALS}")
    
    launch_study(NB_TRIALS, STTUDY_NAME)
    
    print("Number of finished trials: {}".format(len(study.trials)))