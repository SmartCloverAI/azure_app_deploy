"""

This is the universal model file!

"""

import json
import pickle
import numpy as np
import pandas as pd

import azureml



def init(model_file=None, **kwargs):
 
  assert model_file is not None, "Model file is required"
  
  with open(model_file, "rb") as f:
    model = pickle.load(f)
  return model

def preprocess(data):
  if isinstance(data, str):
    data = json.loads(data)
    
  if isinstance(data, dict):
    columns = data["input_data"]["columns"]
    index = data["input_data"]["index"]
    data = data["input_data"]["data"]
    df = pd.DataFrame(data, columns=columns, index=index)
    
  elif isinstance(data, pd.DataFrame):
    df = data  
  return df
    
    
def run(data, model):
  df = preprocess(data)
  if hasattr(model, "forecast"):
    result = model.forecast(df)  
  elif hasattr(model, "predict"):
    result = model.predict(df)
  else:
    msg = f"Model {model} does not have a predict method"
    raise ValueError(msg)
  if isinstance(result, np.ndarray):
    return result.tolist()
  else:
    return result[0].tolist()
