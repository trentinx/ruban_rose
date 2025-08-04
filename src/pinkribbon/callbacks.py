"""
Module de callbacks personnalisés pour l'optimisation Optuna avec Keras.

Ce module fournit des callbacks spécialisés pour l'intégration d'Optuna
avec l'entraînement de modèles Keras, incluant le pruning automatique
des essais sous-performants et des stratégies d'early stopping optimisées.

Classes:
    OptunaPruningCallback: Callback pour le pruning automatique des essais Optuna
    OptunaEarlyStopping: Early stopping optimisé pour Optuna
    OptunaReduceLROnPlateau: Réduction du learning rate optimisée pour Optuna
"""

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import optuna

class OptunaPruningCallback(Callback):
    """
    Callback Keras personnalisé pour le pruning automatique des essais Optuna.
    
    Ce callback surveille une métrique spécifiée pendant l'entraînement et
    communique les valeurs intermédiaires à Optuna. Si Optuna détermine que
    l'essai actuel est sous-performant comparé aux autres, il déclenche
    l'exception TrialPruned pour arrêter l'entraînement prématurément.
    
    Attributes:
        trial (optuna.Trial): L'objet trial Optuna associé à cet entraînement
        monitor (str): Nom de la métrique à surveiller (ex: 'val_accuracy', 'val_loss')
        mode (str): Mode de surveillance ('max' pour maximiser, 'min' pour minimiser)
    
    Example:
        >>> trial = study.ask()
        >>> callback = OptunaPruningCallback(trial, monitor='val_recall', mode='max')
        >>> model.fit(X, y, callbacks=[callback])
    """
    
    def __init__(self, trial, monitor='val_accuracy', mode='max'):
        """
        Initialise le callback de pruning Optuna.
        
        Args:
            trial (optuna.Trial): L'objet trial Optuna pour cet entraînement
            monitor (str, optional): Métrique à surveiller. Defaults to 'val_accuracy'.
            mode (str, optional): Mode d'optimisation ('max' ou 'min'). Defaults to 'max'.
        """
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        """
        Appelé à la fin de chaque époque pour rapporter la métrique et vérifier le pruning.
        
        Cette méthode extrait la valeur de la métrique surveillée, la rapporte à Optuna,
        puis vérifie si l'essai doit être élagué. Si c'est le cas, elle arrête
        l'entraînement et lève l'exception TrialPruned.
        
        Args:
            epoch (int): Numéro de l'époque actuelle (0-indexé)
            logs (dict, optional): Dictionnaire contenant les métriques de l'époque.
                                 Contient typiquement 'loss', 'val_loss', 'accuracy', etc.
        
        Raises:
            optuna.exceptions.TrialPruned: Si Optuna détermine que l'essai doit être élagué
        """
        # Report intermediate value to Optuna
        reported_value = logs.get(self.monitor)
        if reported_value is None:
            return
        
        self.trial.report(reported_value, epoch)
           
        if self.trial.should_prune():
            print(f"Trial {self.trial.number} pruned at epoch {epoch + 1}")
            self.model.stop_training = True
            raise optuna.exceptions.TrialPruned()
        
class OptunaEarlyStopping(EarlyStopping):
    """
    Callback d'early stopping optimisé pour les études Optuna.
    
    Cette classe hérite d'EarlyStopping de Keras et fournit une configuration
    par défaut optimisée pour les expérimentations Optuna. Elle surveille
    la loss de validation et restaure automatiquement les meilleurs poids
    lorsque l'entraînement est arrêté prématurément.
    
    Attributes:
        patience (int): Nombre d'époques sans amélioration avant arrêt
        monitor (str): Métrique surveillée ('val_loss')
        mode (str): Mode de surveillance ('min' pour minimiser la loss)
        restore_best_weights (bool): Restaure les meilleurs poids (True)
        verbose (int): Niveau de verbosité (1 pour afficher les messages)
    
    Example:
        >>> es_callback = OptunaEarlyStopping(patience=10)
        >>> model.fit(X, y, callbacks=[es_callback])
    """
    
    def __init__(self, patience=7):
        """
        Initialise le callback d'early stopping avec des paramètres optimisés pour Optuna.
        
        Args:
            patience (int, optional): Nombre d'époques à attendre sans amélioration
                                    avant d'arrêter l'entraînement. Defaults to 7.
        """
        super().__init__(patience=patience,  
                         restore_best_weights=True,
                         verbose=1,
                         monitor='val_loss',
                         mode='min'
        )

class OptunaReduceLROnPlateau(ReduceLROnPlateau):
    """
    Callback de réduction du learning rate optimisé pour les études Optuna.
    
    Cette classe hérite de ReduceLROnPlateau de Keras et fournit une configuration
    par défaut optimisée pour les expérimentations Optuna. Elle réduit le learning
    rate de moitié lorsque la loss de validation cesse de s'améliorer.
    
    Attributes:
        factor (float): Facteur de réduction du learning rate (0.5)
        patience (int): Nombre d'époques sans amélioration avant réduction
        min_lr (float): Learning rate minimum (1e-7)
        monitor (str): Métrique surveillée ('val_loss')
        mode (str): Mode de surveillance ('min' pour minimiser la loss)
        verbose (int): Niveau de verbosité (1 pour afficher les messages)
    
    Example:
        >>> lr_callback = OptunaReduceLROnPlateau(patience=5)
        >>> model.fit(X, y, callbacks=[lr_callback])
    """
    
    def __init__(self, patience=4):
        """
        Initialise le callback de réduction du learning rate avec des paramètres optimisés pour Optuna.
        
        Args:
            patience (int, optional): Nombre d'époques à attendre sans amélioration
                                    avant de réduire le learning rate. Defaults to 4.
        """
        super().__init__(factor=0.5,
                         patience=patience,
                         min_lr=1e-7,
                         verbose=1,
                         monitor='val_loss',
                         mode='min'
        )
        
__all__ = ["OptunaEarlyStopping", "OptunaPruningCallback", "OptunaReduceLROnPlateau"]