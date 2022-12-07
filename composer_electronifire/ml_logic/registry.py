import glob
import os
import time
import json
from colorama import Fore, Style
from tensorflow.keras import Model, models

def save_model(model: Model=None,
               params: dict=None,
               metrics: dict=None,
               composer: str=None,
               local_registry_path=None) -> None:
    """
    persist trained model, params and metrics
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(local_registry_path, composer)

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)
    # create directory
    if local_registry_path is not None:
        try:
            os.makedirs(os.path.join(local_registry_path, composer))
        except FileExistsError:
            pass
    # save params
    if params is not None:
        params_path = os.path.join(save_path, "params", timestamp + ".json")
        print(f"- params path: {params_path}")
        try:
            os.makedirs(os.path.join(save_path, "params"))
        except FileExistsError:
            pass
        with open(params_path, "w") as file:
            json.dump(params, file)
    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(save_path, "metrics", timestamp + ".json")
        print(f"- metrics path: {metrics_path}")
        try:
            os.makedirs(os.path.join(save_path, "metrics"))
        except FileExistsError:
            pass
        with open(metrics_path, "w") as file:
            json.dump(metrics, file)
    # save model
    if model is not None:
        model_path = os.path.join(save_path, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)
    print("\n:weißes_häkchen: data saved locally")
    return None

def load_model(custom_objects: dict=None,
               save_copy_locally=False,
               composer: str=None,
               local_registry_path=None) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)
    # get latest model version
    model_directory = os.path.join(local_registry_path, composer, "models")
    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None
    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")
    model = models.load_model(model_path, custom_objects=custom_objects)
    print("\n:weißes_häkchen: model loaded from disk")
    return model

def save_midi(midi,
              composer: str=None,
              local_registry_path=None) -> None:
    """Save predicted midi to predictions directory in local registry"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(local_registry_path, composer)
    try:
        os.makedirs(os.path.join(save_path, "predictions"))
    except FileExistsError:
        pass
    midi.write(os.path.join(save_path, "predictions", timestamp)+'.mid')
    print("\n:weißes_häkchen: midi saved locally")
    return None
