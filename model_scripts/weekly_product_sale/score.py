import pickle
import numpy as np
import azureml
import json
import pandas as pd


def init():
  global model
  global COLUMNS

  COLUMNS = [
    "ID_ARTICOL", "ID_CATEGORIE", "ID_FIRMA", "START_DATE_WEEK", "PRET_UNITAR", "DENUMIRE_PRODUCATOR", 
    "PROC_TVA", "PROCENT_DISCOUNT", "SAPTAMANA_W1", "SAPTAMANA_W1_W4", "SAPTAMANA_W1_W12", 
    "SAPTAMANA_W1_W52", "SAPTAMANA_W53", "SAPTAMANA_W53_W56", "SAPTAMANA_W53_W64", 
    "SAPTAMANA_W53_W104", "STOC_W1", "STOC_W2", "STOC_M1"
  ]
  
  MODEL = "models/mflow_weekly_product_sale/model.pkl"
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
  
  data = """
  {
    "input_data": {
      "columns": [
          "ID_ARTICOL", "ID_CATEGORIE", "ID_FIRMA", "START_DATE_WEEK", "PRET_UNITAR", "DENUMIRE_PRODUCATOR", 
          "PROC_TVA", "PROCENT_DISCOUNT", "SAPTAMANA_W1", "SAPTAMANA_W1_W4", "SAPTAMANA_W1_W12", 
          "SAPTAMANA_W1_W52", "SAPTAMANA_W53", "SAPTAMANA_W53_W56", "SAPTAMANA_W53_W64", 
          "SAPTAMANA_W53_W104", "STOC_W1", "STOC_W2", "STOC_M1"
      ],
      "index": [0,1],
      "data": [
        [86,null,114,"2024-10-07 00:00:00.000",22.18,"PRODUCATOR12",5,0,2238,9066,27646,128204,2359,9330,28185,119764,434,425,434],
        [87,null,114,"2024-10-07 00:00:00.000",24.07,"PRODUCATOR2",19,0,2034,7786,23594,112613,2228,8836,26785,129511,287,300,287]
      ]
    }
  }  
  """
  
    
  endpoint_data = """
  {
    "idArticol":104,"idCategorie":19,"idFirma":114,
    "startDateWeek":"2024-12-02T00:00:00",
    "pretUnitar":30.76,"denumireProducator":"PRODUCATOR14","procTVA":19,"procentDiscount":21,"saptamanaW1":2782,"saptamanaW1W4":9341,"saptamanaW1W12":26907,"saptamanaW1W52":117208,"saptamanaW53":2472,"saptamanaW53W56":8649,"saptamanaW53W64":25682,"saptamanaW53W104":115452,"stocW1":285,"stocW2":393,"stocM1":285,"saptamanaW2":2178,"saptamanaW3":2162,"saptamanaW4":2219,"saptamanaW5":2152,"saptamanaW6":2270,"saptamanaW7":2141,"saptamanaW8":2136,"saptamanaW9":2220,"saptamanaW10":2128,"saptamanaW11":2262,"saptamanaW12":2257} 
  """
  endpoint_data = json.loads(endpoint_data)
  
  expected = [2593.479736328125]
  
  
  data = from_endpoint(endpoint_data)

  
  result = run(data)
  print("Expected: ", expected)
  print("Model:    ", result)
  
  assert result == expected
  
  