from keras.applications import ResNet50
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD


from pinkribbon.dataset import DataGenerator

import optuna

class OptunaPruningCallback(Callback):
    """Callback to prune unpromising trials with Optuna."""
    
    def __init__(self, trial, monitor='val_loss', mode='min'):
        super(OptunaPruningCallback, self).__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        # Report intermediate value to Optuna
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Pour les métriques à maximiser, on rapporte la valeur négative
        # pour que le pruner MedianPruner fonctionne correctement
        if self.mode == 'max':
            reported_value = -current_value
        else:
            reported_value = current_value
            
        self.trial.report(reported_value, epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            print(f"Trial {self.trial.number} pruned at epoch {epoch + 1}")
            self.model.stop_training = True

def build_model(dropout_rate):    
    resnet = ResNet50( include_top=False,
                       weights="imagenet",
                       input_shape=(50,50,3),
                       name="resnet50")
    resnet.trainable = False
    
    model = Sequential()
    model.add(resnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout_rate))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

    

# 1. Define an objective function to be maximized.
def objective(trial):
    
    EPOCHS = 10
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        optimizer = SGD(learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = Adam( learning_rate=lr, weight_decay=weight_decay)
    
    generator = DataGenerator("data/BHI.zip")
    sample_generator = generator.sample(1280, shuffle=True)
    train_generator = sample_generator.sample(0.8)
    train_generator.batch_size = batch_size
    test_generator = sample_generator.sample(0.2,reverse=True)
    
    
    es = EarlyStopping( patience=10, 
                        min_delta=0.01,
                        monitor="val_recall",
                        mode='max',
                        verbose=0)
    
    rp = ReduceLROnPlateau( factor=0.5,
                            cooldown=5,
                            patience=5,
                            min_lr=1e-4,
                            min_delta=0.01,
                            monitor="val_recall",
                            mode='max',
                            verbose=0)
    
    cp = ModelCheckpoint( f"results/models/resnet50/trial_{trial.number}.keras",
                          save_best_only=True,
                          monitor="val_recall",
                          mode="max",
                          verbose=0)
    
    # Callback pour le pruning Optuna
    pc = OptunaPruningCallback(trial, monitor='val_recall', mode='max')

    clear_session()
    model = build_model(dropout_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['recall'])
    
    try:
        model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[es, rp, cp, pc]
        )
    except optuna.TrialPruned:
        # Si l'essai est élagué, on retourne une valeur élevée
        raise optuna.TrialPruned()

    score = model.evaluate(test_generator, verbose=0)
    return score[1]

if __name__ == "__main__":
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    NB_TRIALS = 10
   
    # Création d'un pruner MedianPruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Nombre d'essais avant de commencer le pruning
        n_warmup_steps=3,    # Nombre d'époques avant de commencer le pruning
        interval_steps=1     # Intervalle d'évaluation pour le pruning
    )
    
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///results/optuna.db',
        study_name='resnet_50',
        load_if_exists=True,
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=NB_TRIALS)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))