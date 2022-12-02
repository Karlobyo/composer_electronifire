from composer_electronifire.ml_logic.data import load_stream, extract_notes, clean_data
from composer_electronifire.ml_logic.preprocessor import note_transformer
from composer_electronifire.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from composer_electronifire.ml_logic.registry import save_model
from composer_electronifire.ml_logic.params import DATA_SOURCE, LOCAL_DATA_PATH, PROJECT_NAME, DATASET_NAME, FEATURES_NAME, TARGET_NAME
from composer_electronifire.data_sources.big_query import get_bq_data
from composer_electronifire.data_sources.local import get_local_data

import numpy as np
import pandas as pd

def run_model_training():

###### For datasource local!

    if DATA_SOURCE == "local":
        # Get X
        X_path = f"{LOCAL_DATA_PATH}/{FEATURES_NAME}.csv"
        X = get_local_data(X_path)
        X = np.array(X)
        X = np.expand_dims(X, axis=2)

        # Get y
        y_path = f"{LOCAL_DATA_PATH}/{TARGET_NAME}.csv"
        y = get_local_data(y_path)
        y = np.array(y)

        # Initialize model
        model = initialize_model(X, y)

        # Compile model
        model = compile_model(model)

        # Train model
        model, history = train_model(model=model,
                                     X=X,
                                     y=y,
                                     )
        # Turn history into dict
        metrics = dict(accuracy=np.max(history.history['accuracy']),
                       val_accuracy=np.max(history.history['val_accuracy'])
                       )
        print(metrics)
        # Save model
        save_model(model=model,
                   params=None,
                   metrics=metrics
                   )

##### For datasource bigquery!

    if DATA_SOURCE == "bigquery":
        # Get X
        X_table = f"{PROJECT_NAME}.{DATASET_NAME}.{FEATURES_NAME}"
        X = get_bq_data(X_table)
        X = np.array(X)
        X = np.expand_dims(X, axis=2)

        # Get y
        y_table = f"{PROJECT_NAME}.{DATASET_NAME}.{TARGET_NAME}"
        y = get_bq_data(y_table)
        y = np.array(y)

        # Initialize model
        model = initialize_model(X, y)

        # Compile model
        model = compile_model(model)

        # Train model
        model, history = train_model(model=model,
                                     X=X,
                                     y=y,
                                     )

        # Turn history into dict
        metrics = dict(accuracy=np.max(history.history['accuracy']),
                       val_accuracy=np.max(history.history['val_accuracy'])
                       )

        # Save model
        save_model(model=model,
                   params=None,
                   metrics=metrics
                   )


if __name__ == '__main__':
    run_model_training()
