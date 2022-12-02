"""music_generator model package params
load and validate the environment variables in the `.env`
"""
import os
import numpy as np
#DATASET_SIZE = os.environ.get("DATASET_SIZE")
#VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
SEQ_LENGTH = int(os.environ.get("SEQ_LENGTH"))

LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")
DATA_SOURCE = os.environ.get("DATA_SOURCE")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
DATASET_NAME = os.environ.get("DATASET_NAME")
FEATURES_NAME = os.environ.get("FEATURES_NAME")
TARGET_NAME = os.environ.get("TARGET_NAME")
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