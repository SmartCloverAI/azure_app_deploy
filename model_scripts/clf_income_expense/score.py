import pickle
import numpy as np
import azureml
import json
import pandas as pd


def init():
  global model
  global COLUMNS

  COLUMNS = [
          "ID_FIRMA", "ID_TRANZACTIE_NC", "ID_TRANZACTIE", "ID_NC", "DATA_OPERARE", "DEBIT_CREDIT", 
          "SUMA", "ID_ARTICOL", "ESTE_ANALITIC", "ESTE_INACTIV", "ID_UTILIZATOR_ADAUGARE", 
          "DATA_ADAUGARE", "ID_UTILIZATOR_MODIFICARE", "DATA_MODIFICARE"
  ]
  
  MODEL = "models/mflow_clf_income_expenditure/model.pkl"
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
          "ID_FIRMA", "ID_TRANZACTIE_NC", "ID_TRANZACTIE", "ID_NC", "DATA_OPERARE", "DEBIT_CREDIT", 
          "SUMA", "ID_ARTICOL", "ESTE_ANALITIC", "ESTE_INACTIV", "ID_UTILIZATOR_ADAUGARE", 
          "DATA_ADAUGARE", "ID_UTILIZATOR_MODIFICARE", "DATA_MODIFICARE"
      ],
      "index": [0],
      "data": [
          [114, 89, 55, 1, "2022-09-06", "C", 80.40, "null", 0, 0, 56, "2022-09-02 13:50:08.900", 56, "2022-09-02 13:50:08.900"]
      ]
  }
  }  
  """
  str_endpoint_data = """
  [
    {"idFirma":114,"idTranzactieNc":2486,"idTranzactie":1098,"idNc":1,"dataOperare":"2021-12-31T00:00:00.000Z","debitCredit":"D","suma":0,"idArticol":0,"esteAnalitic":false,"esteInactiv":false,"idUtilizatorAdaugare":2094,"dataAdaugare":"2023-01-30T16:21:34.337Z","idUtilizatorModificare":2094,"dataModificare":"2023-01-30T16:21:34.337Z","venitCheltuiala":null},
    {"idFirma":114,"idTranzactieNc":2491,"idTranzactie":1098,"idNc":2,"dataOperare":"2021-12-31T00:00:00.000Z","debitCredit":"C","suma":0,"idArticol":0,"esteAnalitic":false,"esteInactiv":false,"idUtilizatorAdaugare":2094,"dataAdaugare":"2023-01-30T16:21:34.337Z","idUtilizatorModificare":2094,"dataModificare":"2023-01-30T16:21:34.337Z","venitCheltuiala":null},
    {"idFirma":114,"idTranzactieNc":2499,"idTranzactie":1116,"idNc":1,"dataOperare":"2023-01-30T00:00:00.000Z","debitCredit":"C","suma":50000,"idArticol":0,"esteAnalitic":false,"esteInactiv":false,"idUtilizatorAdaugare":2094,"dataAdaugare":"2023-01-30T16:21:34.337Z","idUtilizatorModificare":2094,"dataModificare":"2023-01-30T16:21:34.337Z","venitCheltuiala":null}
  ]
  """

  endpoint_data = json.loads(str_endpoint_data)
  
  endpoint_data_expected = ["CHELTUIALA"]  
  
  data = from_endpoint(endpoint_data[0])
  
  result = run(data)
  print("Expected: ", endpoint_data_expected)
  print("Model:    ", result)
  
  assert result == endpoint_data_expected
  

  endpoint_data_expected = ["VENIT"]  
  
  data = from_endpoint(endpoint_data[1])
  
  result = run(data)
  print("Expected: ", endpoint_data_expected)
  print("Model:    ", result)
  
  assert result == endpoint_data_expected
  