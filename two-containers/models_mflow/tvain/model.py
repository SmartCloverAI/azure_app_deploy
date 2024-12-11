import json
import pickle
import pandas as pd

import azureml


def init(model_file=None, **kwargs):
  global model
  global COLUMNS
  
  assert model_file is not None, "Model file is required"

  COLUMNS = [
    'DENUMIRE_FIRMA', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W1_W52', 'SAPTAMANA_W53', 'SAPTAMANA_W53_W104', 'SAPTAMANA_W1_W12', 'SAPTAMANA_W53_W64', 'ID_FIRMA', 'START_DATE_WEEK', 'SAPTAMANA_W1', 'SAPTAMANA_W53_W56'
  ]
  
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
  
  df = df[COLUMNS]
  return df
    
def run(data):
  df = preprocess(data)
  result = model.forecast(df)
  return result[0].tolist()
  