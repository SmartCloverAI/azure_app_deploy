import json
import pickle
import pandas as pd

import azureml

from utils import log_with_color
from config import APP_NAME


def init():
  global model
  global COLUMNS

  COLUMNS = [
    'DENUMIRE_FIRMA', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W1_W52', 'SAPTAMANA_W53', 'SAPTAMANA_W53_W104', 'SAPTAMANA_W1_W12', 'SAPTAMANA_W53_W64', 'ID_FIRMA', 'START_DATE_WEEK', 'SAPTAMANA_W1', 'SAPTAMANA_W53_W56'
  ]
  
  MODEL = f"./models/{APP_NAME}/model.pkl"
  log_with_color(f"Loading model from {MODEL}", color="y")
  with open(MODEL, "rb") as f:
    model = pickle.load(f)
  log_with_color("Model loaded", color="g")
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
  log_with_color("Running preprocess", color="d")
  df = preprocess(data)
  log_with_color("Running forecast", color="d")
  result = model.forecast(df)
  return result[0].tolist()
  