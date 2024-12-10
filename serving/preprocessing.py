import numpy as np
import pandas as pd


def preprocess_custom(request_data):
  # Expecting a JSON with a single "data" key
  data = request_data.get('data')
  if data is None:
    raise ValueError("The request JSON must contain a 'data' key.")
  data = np.array(data)
  return data

def preprocess_azureml(request_data, input_columns):
  # Expecting a specific JSON structure for AzureML models
  input_data = request_data.get('input_data')
  if input_data is None:
    raise ValueError("The request JSON must contain an 'input_data' key.")
  
  columns = input_data.get('columns')
  data = input_data.get('data')
  index = input_data.get('index')
  
  if columns is None or data is None or index is None:
    raise ValueError("The 'input_data' must contain 'columns', 'data', and 'index' keys.")
  
  df = pd.DataFrame(data, index=index, columns=columns)
  if input_columns is not None:
    df = df[input_columns]
  return df
