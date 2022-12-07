from composer_electronifire.ml_logic.data import load_stream,\
                                                 extract_notes,\
                                                 clean_data,\
                                                 midis_set,\
                                                 midi_to_notes,\
                                                 notes_to_midi,\
                                                 notes_to_chords,\
                                                 strongest_note,\
                                                 join_dfs
                                                 
from composer_electronifire.ml_logic.preprocessor import note_transformer,\
                                                         data_split,\
                                                         df_to_dataset,\
                                                         create_sequences,\
                                                         create_batches
                                                         
from composer_electronifire.ml_logic.model import initialize_model,\
                                                  compile_model,\
                                                  train_model,\
                                                  evaluate_model,\
                                                  initialize_mv_model,\
                                                  mse_with_positive_pressure,\
                                                  compile_mv_model,\
                                                  train_mv_model,\
                                                  predict_notes
                                                  
from composer_electronifire.ml_logic.registry import save_model,\
                                                     load_model,\
                                                     save_midi
                                                     
from composer_electronifire.ml_logic.params import BATCH_SIZE,\
                                                   SEQ_LENGTH,\
                                                   EPOCHS,\
                                                   PATIENCE,\
                                                   CALLBACKS,\
                                                   SHIFT,\
                                                   COLUMNS,\
                                                   NUM_PREDICTIONS,\
                                                   DATA_SOURCE,\
                                                   LOCAL_DATA_PATH,\
                                                   LOCAL_MIDI_PATH,\
                                                   LOCAL_REGISTRY_PATH,\
                                                   PROJECT_NAME,\
                                                   DATASET_NAME,\
                                                   FEATURES_NAME,\
                                                   TARGET_NAME,\
                                                   MV_TRAIN_DF,\
                                                   MV_VAL_DF,\
                                                   MV_SEED_DF
                                                
from composer_electronifire.data_sources.big_query import get_bq_data
from composer_electronifire.data_sources.local import get_local_data

import pretty_midi as pm
import numpy as np
import pandas as pd
import os

###### Baseline model

def run_model_training(source_type=DATA_SOURCE):

    if source_type == "local":      # For datasource local!
        # Get X
        X_path = f"{LOCAL_DATA_PATH}/{FEATURES_NAME}.csv"
        X = get_local_data(X_path)
        X = np.array(X)
        X = np.expand_dims(X, axis=2)
        # Get y
        y_path = f"{LOCAL_DATA_PATH}/{TARGET_NAME}.csv"
        y = get_local_data(y_path)
        y = np.array(y)

    if source_type == "bigquery":   # For datasource bigquery!
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
                                     batch_size=BATCH_SIZE
                                     )
        # Turn params into dict
        params = dict(batch_size=BATCH_SIZE)

        # Turn history into dict
        metrics = dict(accuracy=np.max(history.history['accuracy']),
                       val_accuracy=np.max(history.history['val_accuracy'])
                       )
        print(metrics)
        # Save model
        save_model(model=model,
                   params=params,
                   metrics=metrics,
                   local_registry_path=LOCAL_REGISTRY_PATH
                   )

##### Multivariate model

def preprocess_mv_datasets():
    midi_lst = midis_set(datapath=LOCAL_MIDI_PATH)
    df_lst = [midi_to_notes(midi=midi) for midi in midi_lst]
    df_lst = [strongest_note(df=df) for df in df_lst]
    df = join_dfs(midi_lst)
    train_df, val_df, seed_df = data_split(df, seq_length=SEQ_LENGTH)
    train_df.to_csv(os.path.join(LOCAL_DATA_PATH,MV_TRAIN_DF)+'.csv')
    val_df.to_csv(os.path.join(LOCAL_DATA_PATH,MV_VAL_DF)+'.csv')
    seed_df.to_csv(os.path.join(LOCAL_DATA_PATH,MV_SEED_DF)+'.csv')
    print(f"Preprocessed midis at {LOCAL_MIDI_PATH}")
    return None

def run_mv_model_training():
    model = initialize_mv_model(seq_length=SEQ_LENGTH, cols=COLUMNS)
    model = compile_mv_model(model, cols=COLUMNS)
    
    train_df = pd.read_csv(os.path.join(LOCAL_DATA_PATH,MV_TRAIN_DF)+'.csv')
    val_df = pd.read_csv(os.path.join(LOCAL_DATA_PATH,MV_VAL_DF)+'.csv')
    seed_df = pd.read_csv(os.path.join(LOCAL_DATA_PATH,MV_SEED_DF)+'.csv')
    
    train_ds = df_to_dataset(train_df, cols=COLUMNS)
    val_ds = df_to_dataset(val_df, cols=COLUMNS)
    seed_ds = df_to_dataset(seed_df, cols=COLUMNS)
    
    train_seq = create_sequences(train_ds, seq_length=SEQ_LENGTH, cols=COLUMNS, shift=SHIFT)
    val_seq = create_sequences(val_ds, seq_length=SEQ_LENGTH, cols=COLUMNS, shift=SHIFT)
    seed_seq = create_sequences(seed_ds, seq_length=SEQ_LENGTH, cols=COLUMNS, shift=SHIFT)
    
    train_ds = create_batches(train_seq, batch_size=BATCH_SIZE)
    val_ds = create_batches(val_seq, batch_size=BATCH_SIZE)
    seed_ds = create_batches(seed_seq, batch_size=BATCH_SIZE)

    model, history = train_mv_model(model=model,
                                    dataset=train_ds,
                                    validation_data=val_ds,
                                    epochs=EPOCHS,
                                    patience=PATIENCE,
                                    callbacks=CALLBACKS,
                                    local_registry_path=LOCAL_REGISTRY_PATH)

    params = dict(batch_Size=str(BATCH_SIZE), epochs=str(EPOCHS))
    save_model(model=model,
               params=params,
               metrics=history.history,
               local_registry_path=LOCAL_REGISTRY_PATH
               )
    return None

def predict_mv_model(notes_path: str=os.path.join(LOCAL_DATA_PATH,MV_SEED_DF)+'.csv',
                     input: str='df',
                     local_registry_path: str=LOCAL_REGISTRY_PATH):
    """Generate midi as note prediction from model for notes DF or Midi
    Parameters:
    notes_path: Path to a .csv dtaframe or .mid midi file
    input: choose method 'dataframe' or 'midi'
    local_registry_path: Path to save midi file"""

    model = load_model(local_registry_path=local_registry_path,
                       custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure}
                       )
    if input == 'path':
        midi = pm.PrettyMIDI(notes_path)
        notes = midi_to_notes(midi=midi)
    elif input == 'df':
        notes = pd.read_csv(notes_path)
    else:
        return print("Select input - 'notes' or 'path' !")

    gen_df = predict_notes(notes=notes, 
                           model=model,
                           num_predictions=NUM_PREDICTIONS,
                           seq_length=SEQ_LENGTH,
                           cols=COLUMNS
                           )
    gen_midi = notes_to_midi(gen_df, 
                             cols=COLUMNS)
    save_midi(midi=gen_midi, local_registry_path=LOCAL_REGISTRY_PATH)
    print("\nSaved predicted midi")
    return None

if __name__ == '__main__':
    run_model_training()
    preprocess_mv_datasets()
    run_mv_model_training()
    predict_mv_model()
