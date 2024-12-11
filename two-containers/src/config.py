"""
This module defines the configuration handling for loading models.
"""

import json
import os
import importlib

from utils import log_with_color


class ModelLoader:
  """
  A class to load a model dynamically from a given model name and module file.

  Parameters
  ----------
  model_name : str
    The name of the model folder to load.
    
  script_file : str
    The name of the python file (e.g. 'model.py', 'score.py') where init() and run() are defined.
    
  model_file : str
    The path to the model file to load within the `init` function (will be passed as `model_file`).

  Attributes
  ----------
  model_name : str
    The name of the model.
  script_file : str
    The filename of the module to import (should contain init and run).
  model_module : module
    The dynamically imported model module.
  init_fn : callable
    The init() function from the model module.
  run_fn : callable
    The run() function from the model module.
  """

  def __init__(self, model_name: str, script_file: str = "model.py", model_file: str = None):
    log_with_color(f"Loading model {model_name} from {script_file}", color="b")
    self.model_name = model_name
    self.script_file = script_file
    self.model_module = None
    self.model_file = model_file
    self.init_fn = None
    self.run_fn = None
    self._load_model()
    return

  def _load_model(self):
    """
    Dynamically import the specified module file from the corresponding model folder
    and bind the init and run functions.
    """
    model_path = os.path.splitext(self.script_file)[0]  # remove .py if present
    # Remove leading './' if present
    if model_path.startswith("./"):
        model_path = model_path[2:]
    # Remove leading '/' if present
    elif model_path.startswith("/"):
        model_path = model_path[1:]

    # Replace '/' with '.' to get the module import path
    # e.g. "models/something/script" -> "models.something.script"
    #      "model/scripts/test" -> "model.scripts.test"
    model_path = model_path.replace("/", ".")
    
    log_with_color(f"Loading script {model_path} from {self.script_file}", color="b")    
    self.model_module = importlib.import_module(model_path)
    if not hasattr(self.model_module, "init") or not hasattr(self.model_module, "run"):
      raise ValueError(f"Module {self.script_file} in model {self.model_name} must define init() and run() functions.")
    self.init_fn = self.model_module.init
    self.run_fn = self.model_module.run
    log_with_color(f"Loaded model {self.model_name} from {self.script_file}", color="g")
    return

  def init(self):
    """
    Initialize the model by calling its init function.
    """
    log_with_color(f"Initializing model <{self.model_name}> with {self.model_file}", color="b")
    self.init_fn(model_file=self.model_file)
    log_with_color(f"Initialized model <{self.model_name}>.", color="g")
    return

  def run(self, request_data):
    """
    Run inference on the given request_data using the model's run function.

    Parameters
    ----------
    request_data : dict
      The input data to be passed to the model's run function.

    Returns
    -------
    Any
      The result of the model's run function.
    """
    log_with_color(f"Running model {self.model_name}", color="d")
    return self.run_fn(request_data)


class ModelsHandler:
  """
  A class that holds multiple ModelLoader instances.

  Parameters
  ----------
  config_path : str
    The path to the config.json file.

  Attributes
  ----------
  models : dict
    A dictionary mapping model_name -> ModelLoader instance.
  """

  def __init__(self, config_path: str):
    self.models = {}
    self._load_config(config_path)
    return

  def _load_config(self, config_path: str):
    """
    Load the configuration file and initialize models.

    Parameters
    ----------
    config_path : str
      The path to the config.json file.
    """
    log_with_color(f"Loading models from {config_path}", color="b")
    if not os.path.exists(config_path):
      raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
      config_data = json.load(f)
      
    log_with_color(f"Loaded config file:\n {json.dumps(config_data, indent=2)}", color="d")

    for model_info in config_data:
      model_name = model_info.get("name")
      folder = model_info.get("folder")
      if model_name is None:
        continue
      script_file = model_info.get("script_file", "model.py")
      model_file = model_info.get("model_file", None)
      if model_file is not None:
        # check if full path
        if not os.path.exists(model_file):
          model_file = os.path.join(folder, model_file)
        
        # now assert that the file exists
        if not os.path.exists(model_file):
          raise FileNotFoundError(f"Model file {model_file} not found.")
      
      script_file = os.path.join(folder, script_file)

      loader = ModelLoader(
        model_name=model_name, script_file=script_file, model_file=model_file
      )
      loader.init()
      self.models[model_name] = loader
    return

  def get_model(self, model_name: str):
    """
    Retrieve a loaded model by name.

    Parameters
    ----------
    model_name : str
      The name of the model to retrieve.

    Returns
    -------
    ModelLoader
      The corresponding ModelLoader instance.

    Raises
    ------
    KeyError
      If the model_name is not found.
    """
    if model_name not in self.models:
      raise KeyError(f"No model named {model_name} is loaded.")
    return self.models[model_name]
