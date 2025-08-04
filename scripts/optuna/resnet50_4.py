"""
Script d'optimisation d'hyperparamètres pour modèle ResNet50 avec Optuna.

Ce script implémente un pipeline d'optimisation multi-objectifs pour un modèle
de classification binaire basé sur ResNet50 pré-entraîné. Il utilise Optuna
pour optimiser simultanément le recall (à maximiser) et la loss (à minimiser).

Le modèle utilise ResNet50 comme extracteur de caractéristiques gelé, suivi
de couches denses personnalisées pour la classification binaire avec sigmoid.

Sur la base de l'essai #19 de l'expérience resnet_2, on cherche  à déterminer 
le nombre de couches optimal à dégeler près de la tête de resnet afin d'améliorer 
l'atteinte des onjectifs fixés.

Prétraitement appliqué:
Images en 224x224x3 en RGB + CLAHE + préprocessing resnet.

Usage:
    python resnet50_4.py

Dependencies:
    - keras (TensorFlow)
    - optuna
    - pinkribbon (module local)
    - numpy, pandas (via dependencies)
"""

from keras.backend import clear_session
from keras.optimizers import Adam
from pinkribbon.callbacks import *
from pinkribbon.dataset import DataGenerator
from pinkribbon.models import build_model_resnet50_4 as build_model
import gc
import optuna


NB_TRIALS = 20
STUDY_NAME = "resnet50_5"




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
    
    # Study resnet50_2 , trial 19
    lr = 0.0007307590991963969
    batch_size = 16
    dropout_rate = 0.4464874858615145
    weight_decay = 1.794315918374775e-06
    optimizer = Adam(learning_rate=lr, weight_decay=weight_decay)
    unfrozen  = trial.suggest_int('unfrozen', 1, 40)

    train_generator.batch_size = batch_size
    val_generator.batch_size = batch_size

    clear_session()
    gc.collect()
    model = build_model(dropout_rate, unfrozen)
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
        study_name=STUDY_NAME,
        load_if_exists=True,
        pruner=pruner
    )
    

    
    print(f"Démarrage de l'optimisation avec l'étude: {STUDY_NAME}")
    print(f"Nombre d'essais prévus: {NB_TRIALS}")
    
    launch_study(NB_TRIALS, STUDY_NAME)
    
    print("Number of finished trials: {}".format(len(study.trials)))