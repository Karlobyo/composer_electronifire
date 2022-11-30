from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from typing import Tuple

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

def compile_model(model):
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
    #Training the Model
    es = EarlyStopping(patience=patience,
                       restore_best_weights=True,
                       verbose=0)
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
