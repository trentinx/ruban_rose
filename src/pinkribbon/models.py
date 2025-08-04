from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.applications import ResNet50
from .callbacks import OptunaEarlyStopping, OptunaReduceLROnPlateau
from .dataset import DataGenerator


def build_model_resnet50_1(dropout_rate, unfrozen):
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

def build_model_resnet50_2(dropout_rate, unfrozen):
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


def build_model_resnet50_3(dropout_rate, unfrozen):
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

def build_model_resnet50_4(dropout_rate, unfrozen):
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
                       pooling = "avg",
                       name="resnet50")
    for layer in resnet.layers[:-unfrozen]:
        layer.trainable = False
    for layer in resnet.layers[-unfrozen:]:
        layer.trainable = True 
    
    model = Sequential()
    model.add(resnet)
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def retrain_resnet50(df,id, study_name, build_function):
    EPOCHS=15
    parameters = df.iloc[id].to_dict()
    generator = DataGenerator("data/BHI.zip", max_samples_per_class=1500, seed=42, experimentation=study_name)
    train_generator, val_generator = generator.train_test_split(0.2)
    del generator
    val_generator.batch_size = 1
    train_generator.shuffle = True
    
    print(f"Train data size : {len(train_generator) * train_generator.batch_size}")
    print(f"Test data size : {len(val_generator) * val_generator.batch_size}")

    if "optimizer" in parameters:
        if parameters["optimizer"] == "SGD":
            optimizer = SGD(learning_rate=parameters["learning_rate"], momentum=parameters["momentum"], weight_decay=parameters["weight_decay"])
        else:
            optimizer = Adam(learning_rate=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
    else:
        optimizer = Adam(learning_rate=0.0007307590991963969, weight_decay=1.794315918374775e-06)
    
    train_generator.batch_size = parameters.get("batch_size",32)
    val_generator.batch_size = parameters.get("batch_size",32)


    model = build_function(parameters.get("dropout_rate",0.4464874858615145),
                           parameters.get("unfrozen_layers",None))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["recall"])

    es = OptunaEarlyStopping()
    rp = OptunaReduceLROnPlateau()
    
    model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[es, rp]
        )
    return model

__all__ = [ "build_model_resnet50_1", "build_model_resnet50_2", 
           "build_model_resnet50_3", "build_model_resnet50_4",
           "retrain_resnet50"]