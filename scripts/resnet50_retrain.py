from pinkribbon.models import *
import pandas as pd

columns = ["trial","state","recall","loss","batch_size",
           "dropout_rate","learning_rate","momentum","optimizer",
           "weight_decay","history"]

columns =['trial', 'state', 'recall', 'loss', 'unfrozen_layers','history']

if __name__ == "__main__":
    
    STUDY_NAME = "resnet50_5"
    BUILD_FUNCTION = build_model_resnet50_4
    ID = 18
    
    df = pd.read_csv(f"data/{STUDY_NAME}.csv",encoding="utf-8", header=0, names=columns)
    model = retrain_resnet50(df,ID, STUDY_NAME, BUILD_FUNCTION)
    model.save(f"models/{STUDY_NAME}_{ID}.keras")