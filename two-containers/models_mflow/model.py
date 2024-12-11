"""

This is the universal model file!

"""

import json
import pickle
import pandas as pd

import azureml



def init(model_file=None, **kwargs):
  global model
  
  assert model_file is not None, "Model file is required"
  
  with open(model_file, "rb") as f:
    model = pickle.load(f)
  return

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
    
def run(data):
  df = preprocess(data)
  result = model.forecast(df)
  return result[0].tolist()
  