import pickle
import numpy as np
import azureml
import json
import pandas as pd


def init():
  global model
  global COLUMNS

  COLUMNS = [
    'DENUMIRE_FIRMA', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W1_W52', 'SAPTAMANA_W53', 'SAPTAMANA_W53_W104', 'SAPTAMANA_W1_W12', 'SAPTAMANA_W53_W64', 'ID_FIRMA', 'START_DATE_WEEK', 'SAPTAMANA_W1', 'SAPTAMANA_W53_W56'
  ]
  
  MODEL = "models/tvain/model.pkl"
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
  
  endpoint_data = {
  "startDateWeek": "2024-12-02 00:00:00",
  "idFirma": 114,
  "denumireFirma": "CLOUD ACCOUNTING SRL",
  "saptamanaW1": 459088.69,
  "saptamanaW1W4": 1919241.64,
  "saptamanaW1W12": 5766925.09,
  "saptamanaW1W52": 24157681.17,
  "saptamanaW53": 439931.23,
  "saptamanaW53W56": 1836753.4,
  "saptamanaW53W64": 5561803.84,
  "saptamanaW53W104": 23248200.08,
  "saptamanaW2": 486239.27,
  "saptamanaW3": 490614.88,
  "saptamanaW4": 483298.8,
  "saptamanaW5": 480109.04,
  "saptamanaW6": 485780.79,
  "saptamanaW7": 487067.98,
  "saptamanaW8": 482020.77,
  "saptamanaW9": 477480.79,
  "saptamanaW10": 480303.95,
  "saptamanaW11": 479874.81,
  "saptamanaW12": 475045.32
}
  
  data = {
    "input_data": {
      "columns": [
          "START_DATE_WEEK", "ID_FIRMA", "DENUMIRE_FIRMA", 
          "SAPTAMANA_W1", "SAPTAMANA_W1_W4", "SAPTAMANA_W1_W12", 
          "SAPTAMANA_W1_W52", "SAPTAMANA_W53", "SAPTAMANA_W53_W56", "SAPTAMANA_W53_W64", 
          "SAPTAMANA_W53_W104"
      ],
      "index": [0,1],
      "data": [
        ["2024-10-07 00:00:00.000",55,"FIRMA HIGH-TECH SYSTEMS & SOFTWARE SRL",222546.3046,891055.7452,2657789.7205,11017434.3198,211994.7591,848414.7938,2539921.7689,10536265.7779],
        ["2024-10-07 00:00:00.000",114,"CLOUD ACCOUNTING SRL",477480.7863,1912704.8649,5760733.2929,24007120.8383,463158.8105,1858143.9519,5563636.8727,23116338.6095]
      ]
    }
  }
  
  data = from_endpoint(endpoint_data)

  
  result = run(data)
  print(result)
  
  