"""music_generator model package params
load and validate the environment variables in the `.env`
"""
import os
import numpy as np

# Integers
SEQ_LENGTH = int(os.environ.get("SEQ_LENGTH"))
SHIFT = int(os.environ.get("SHIFT"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
EPOCHS = int(os.environ.get("EPOCHS"))
PATIENCE = int(os.environ.get("PATIENCE"))
NUM_PREDICTIONS = int(os.environ.get("NUM_PREDICTIONS"))

# Paths
LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_MIDI_PATH = os.path.expanduser(os.environ.get("LOCAL_MIDI_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))

# Strings
CALLBACKS = os.environ.get("CALLBACKS")
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")
DATA_SOURCE = os.environ.get("DATA_SOURCE")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
DATASET_NAME = os.environ.get("DATASET_NAME")
FEATURES_NAME = os.environ.get("FEATURES_NAME")
TARGET_NAME = os.environ.get("TARGET_NAME")
MV_TRAIN_DF = os.environ.get("MV_TRAIN_DF")
MV_SEED_DF = os.environ.get("MV_SEED_DF")
MV_VAL_DF = os.environ.get("MV_VAL_DF")
COLUMNS = os.environ.get("COLUMNS")
METRICS = os.environ.get("METRICS")
LOSS = os.environ.get("LOSS")


################## VALIDATIONS #################
env_valid_options = dict(
   # DATASET_SIZE=["1k", "10k", "100k", "500k", "50M", "new"],
   # VALIDATION_DATASET_SIZE=["1k", "10k", "100k", "500k", "500k", "new"],
    DATA_SOURCE=["local", "bigquery"],
    MODEL_TARGET=["local", "gcs", "mlflow"],)
def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")
for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
