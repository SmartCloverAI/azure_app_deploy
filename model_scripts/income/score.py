import pickle
import numpy as np
import azureml
import json
import pandas as pd


def init():
  global model
  global COLUMNS

  COLUMNS = [
    'START_DATE_WEEK', 'SAPTAMANA_W53', 'DENUMIRE_FIRMA', 'ID_FIRMA', 'CATEGORIE_VENIT', 
    'SAPTAMANA_W1', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W1_W12', 'SAPTAMANA_W53_W56', 'SAPTAMANA_W53_W104', 
    'SAPTAMANA_W1_W52', 'SAPTAMANA_W53_W64'
  ]
  
  MODEL = "models/income/model.pkl"
  with open(MODEL, "rb") as f:
    model = pickle.load(f)
  print(model)
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
  try:
    df = preprocess(data)
    result = model.forecast(df)
    return result[0].tolist()
  except Exception as e:
    error = str(e)
    return error
  
  
def from_endpoint(request_data):
  data = []
  processed = { x.lower(): request_data[x] for x in request_data}
  for column in COLUMNS:
    _col = column.replace("_", "").lower()
    data.append(processed[_col])
  input_data = {
    "input_data": {
      "columns": COLUMNS,
      "index": [0],
      "data": [data]
    }
  }
  return input_data
  

if __name__ == "__main__":
  
  init()
  
  data = {
    "input_data": {
      "columns": [
          "START_DATE_WEEK", "ID_FIRMA", "DENUMIRE_FIRMA", "CATEGORIE_VENIT","SAPTAMANA_W1", "SAPTAMANA_W1_W4", "SAPTAMANA_W1_W12", 
          "SAPTAMANA_W1_W52", "SAPTAMANA_W53", "SAPTAMANA_W53_W56", "SAPTAMANA_W53_W64", 
          "SAPTAMANA_W53_W104"
      ],
      "index": [0,1],
      "data": [
        ["2024-10-07 00:00:00.000",55,"FIRMA HIGH-TECH SYSTEMS & SOFTWARE SRL","VENITURI",1171296.34,4689767.08,13988366.95,57986496.42,1115761.89,4465341.02,13368009.31,55454030.41],
        ["2024-10-07 00:00:00.000",114,"CLOUD ACCOUNTING SRL","VENITURI",2513056.77,10066867.71,30319648.91,126353267.57,2437677.95,9779705.01,29282299.33,121664940.05]
      ]
    }
  }  
  
  endpoint_data = {
    "startDateWeek": "2024-12-02 00:00:00",
    "idFirma": 114,
    "denumireFirma": "CLOUD ACCOUNTING SRL",
    "categorieVenit": "VENITURI",
    "saptamanaW1": 2416256.24,
    "saptamanaW1W4": 10101271.79,
    "saptamanaW1W12": 30352237.32,
    "saptamanaW1W52": 127145690.39,
    "saptamanaW53": 2315427.54,
    "saptamanaW53W56": 9667123.15,
    "saptamanaW53W64": 29272651.8,
    "saptamanaW53W104": 122358947.79,
    "saptamanaW2": 2559154.06,
    "saptamanaW3": 2582183.59,
    "saptamanaW4": 2543677.9,
    "saptamanaW5": 2526889.71,
    "saptamanaW6": 2556741.02,
    "saptamanaW7": 2563515.67,
    "saptamanaW8": 2536951.42,
    "saptamanaW9": 2513056.77,
    "saptamanaW10": 2527915.52,
    "saptamanaW11": 2525656.87,
    "saptamanaW12": 2500238.55
  }
  
  endpoint_data_expected = [2442085.9741508453]
  
  data = from_endpoint(endpoint_data)

  
  result = run(data)
  print("Expected: ", endpoint_data_expected)
  print("Model:    ", result)
  
  assert result == endpoint_data_expected
  
  