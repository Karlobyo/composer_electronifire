import tensorflow as tf
from tensorflow import Tensor, maximum, reduce_mean, expand_dims, random, squeeze
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import numpy as np
import pandas as pd
from typing import Tuple

### Base model

def initialize_model(X,y) -> Model:
    #Initialising the Model
    model = Sequential()
    #Adding layers
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model

def compile_model(model) -> Model:
    #Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics='accuracy'
                  )
    return model

def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=128,
                patience=20,
                validation_split=0.2,
                validation_data=None) -> Tuple[Model, dict]:
    #Train the Model
    es = EarlyStopping(patience=patience,
                       restore_best_weights=True,
                       verbose=1)
    history = model.fit(X, y,
                        batch_size=batch_size, epochs=150,
                        verbose=0, callbacks=es,
                        validation_split=validation_split
                        )
    return model, history

def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)
    return metrics

### Multivariate model

def initialize_mv_model(seq_length: int=40) -> Model:
    """instantiate RNN model"""

    input_shape = (seq_length, 3)

    inputs = Input(input_shape)
    model = LSTM(512, return_sequences=True)(inputs)

    model = Dropout(0.1)(model)
    model = LSTM(256, return_sequences=True)(model)
    model = Dropout(0.2)(model)
    model = LSTM(128)(model)
    model = Dense(64)(model)
    model = Dropout(0.3)(model)
    outputs = {
        'pitch': Dense(88, name='pitch', activation='softmax')(model),
        'step': Dense(1, name='step')(model),
        'velocity': Dense(1, name='velocity')(model)
    }

    model = Model(inputs, outputs)

    return model

def mse_with_positive_pressure(y_true: Tensor, y_pred: Tensor):
    """Calculate custom mse loss function"""
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * maximum(-y_pred, 0.0)
    return reduce_mean(mse + positive_pressure)

def compile_mv_model(model: Model) -> Model:
    """Compile model with custom loss function"""

    metrics = {'pitch': 'accuracy',
               'step': 'mse',
               'velocity': 'mse'
               }

    loss = {'pitch': SparseCategoricalCrossentropy(from_logits=True),
            'step': mse_with_positive_pressure,
            'velocity': mse_with_positive_pressure
            }

    loss_weights = {'pitch': 0.05,
                    'step': 0.01,
                    'velocity':0.001
                    }

    model.compile(loss=loss,
                  loss_weights=loss_weights,
                  optimizer='rmsprop',
                  metrics=metrics
                  )
    return model

def train_mv_model(model: Model,
                   dataset,
                   validation_data,
                   epochs=500,
                   patience=20,
                   callbacks=None,
                   local_registry_path: str='.'
                   ) -> Tuple[Model, dict]:
    """Fit model with train and validation dataset, """
    mc = ModelCheckpoint(filepath=local_registry_path+'/training_checkpoints/ckpt_{epoch}',
                         save_weights_only=True)
    es =  EarlyStopping(monitor='loss',
                        patience=patience,
                        verbose=0,
                        restore_best_weights=True)

    if callbacks == 'ModelCheckpoint and EarlyStopping':
        callbacks = [mc, es]
    elif callbacks == 'ModelCheckpoint':
        callbacks = [mc]
    elif callbacks == 'EarlyStopping':
        callbacks = [es]
    else:
        callbacks = None

    history = model.fit(dataset,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=validation_data
                        )
    return model, history

def predict_notes(notes: pd.DataFrame,
                  model: Model,
                  temperature: float = 1.0,
                  num_predictions: int=150,
                  seq_length: int=40) -> pd.DataFrame:
    """Generates n following notes as a dataframe."""

    input_notes = (np.stack([notes[key] for key in ['pitch',
                                                    'step',
                                                    'velocity'
                                                    ]], axis=1) - np.array([21, 0, 0]))[-seq_length:]
    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        input_notes = expand_dims(input_notes, 0)
        predictions = model.predict(input_notes, verbose=0)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        velocity = predictions['velocity']

        pitch_logits /= temperature
        pitch = random.categorical(pitch_logits, num_samples=1)
        pitch = squeeze(pitch, axis=-1)
        velocity = squeeze(velocity, axis=-1)
        step = squeeze(step, axis=-1)

        # `step` and `velocity` values should be non-negative
        step = maximum(0, step)
        velocity = maximum(0, velocity)
        input_note = [np.array(pitch)[0], np.array(step)[0], np.array(velocity)[0]]
        generated_notes.append(input_note)
        input_notes = np.delete(np.squeeze(input_notes, 0), 0, axis=0)

        input_notes = np.append(input_notes, np.array(input_note, ndmin=2), axis=0)

    generated_notes = pd.DataFrame(
        generated_notes, columns=['pitch', 'step', 'velocity'])

    return generated_notes
