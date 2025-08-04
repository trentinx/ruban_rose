"""
Module de métriques personnalisées pour l'évaluation de modèles de classification binaire.

Ce module fournit des métriques spécialisées pour l'évaluation de modèles
de classification binaire dans le contexte médical, incluant le coefficient
de corrélation de Matthews (MCC) et des fonctions de perte personnalisées.

Classes:
    MCC: Métrique Keras pour le coefficient de corrélation de Matthews

Fonctions:
    custom_binary_cross_entropy: Fonction de perte binaire cross-entropy personnalisée
"""

from keras.metrics import Metric
import tensorflow as tf
import keras

class MCC(Metric):
    """
    Métrique Keras pour le coefficient de corrélation de Matthews (MCC).
    
    Le MCC est une mesure de qualité pour la classification binaire qui prend
    en compte les vrais et faux positifs et négatifs. Il retourne une valeur
    entre -1 et +1, où +1 représente une prédiction parfaite, 0 une prédiction
    aléatoire, et -1 une prédiction totalement incorrecte.
    
    Cette métrique est particulièrement utile pour les datasets déséquilibrés
    car elle donne une évaluation équilibrée de la performance sur les deux classes.
    
    Attributes:
        tp (tf.Variable): Compteur des vrais positifs
        tn (tf.Variable): Compteur des vrais négatifs  
        fp (tf.Variable): Compteur des faux positifs
        fn (tf.Variable): Compteur des faux négatifs
    
    Formula:
        MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Example:
        >>> mcc_metric = MCC()
        >>> model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mcc_metric])
        >>> model.fit(X_train, y_train, validation_data=(X_val, y_val))
    """
    
    def __init__(self, name='mcc', **kwargs):
        """
        Initialise la métrique MCC.
        
        Args:
            name (str, optional): Nom de la métrique. Defaults to 'mcc'.
            **kwargs: Arguments supplémentaires passés à la classe parent Metric.
        """
        super(MCC, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Met à jour l'état interne de la métrique avec un nouveau batch de données.
        
        Cette méthode calcule les vrais/faux positifs/négatifs pour le batch
        actuel et les ajoute aux compteurs accumulés.
        
        Args:
            y_true (tf.Tensor): Labels réels (0 ou 1) de forme (batch_size,).
            y_pred (tf.Tensor): Prédictions du modèle (probabilités) de forme (batch_size,).
            sample_weight (tf.Tensor, optional): Poids des échantillons. Non utilisé
                                               dans cette implémentation.
        
        Note:
            Les prédictions sont arrondies (seuil à 0.5) pour la classification binaire.
        """
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        """
        Calcule et retourne la valeur finale du coefficient de corrélation de Matthews.
        
        Cette méthode utilise les compteurs accumulés pour calculer le MCC selon
        la formule standard. Elle gère le cas particulier où le dénominateur
        est zéro (retourne 0.0 dans ce cas).
        
        Returns:
            tf.Tensor: Valeur du MCC entre -1 et +1.
                      - +1: Prédiction parfaite
                      -  0: Prédiction aléatoire
                      - -1: Prédiction totalement incorrecte
                      -  0: Cas dégénéré (dénominateur nul)
        """
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        numerator = (tp * tn) - (fp * fn)
        denominator = tf.math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        return tf.where(tf.equal(denominator, 0), 0.0, numerator / denominator)

    def reset_states(self):
        """
        Remet à zéro tous les compteurs internes de la métrique.
        
        Cette méthode est appelée automatiquement au début de chaque époque
        pour réinitialiser les compteurs accumulés.
        """
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


def custom_binary_cross_entropy(y_true, y_pred):
    """
    Fonction de perte binaire cross-entropy personnalisée avec protection numérique.
    
    Cette fonction implémente la perte de cross-entropy binaire avec un clipping
    des prédictions pour éviter les problèmes numériques (log(0) ou log(1)).
    Elle est équivalente à la fonction standard de Keras mais avec un contrôle
    explicite des valeurs limites.
    
    Args:
        y_true (tf.Tensor): Labels réels (0 ou 1) de forme (batch_size,).
        y_pred (tf.Tensor): Prédictions du modèle (probabilités entre 0 et 1)
                           de forme (batch_size,).
    
    Returns:
        tf.Tensor: Valeur scalaire de la perte moyenne sur le batch.
    
    Formula:
        BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    Note:
        - Utilise keras.epsilon() pour éviter log(0)
        - Clip les prédictions entre epsilon et (1 - epsilon)
        - Calcule la moyenne sur tous les échantillons du batch
    
    Example:
        >>> model.compile(optimizer='adam', loss=custom_binary_cross_entropy)
    """
    epsilon = keras.epsilon()  # petite valeur pour éviter log(0)
    y_pred = keras.clip(y_pred, epsilon, 1.0 - epsilon)
    return -keras.mean(y_true * keras.log(y_pred) + (1 - y_true) * keras.log(1 - y_pred))


__all__ = ["MCC", "custom_binary_cross_entropy"]