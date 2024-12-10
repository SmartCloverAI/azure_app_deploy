import numpy as np
import pandas as pd

from serving.preprocessing import preprocess_custom, preprocess_azureml


def score(model, model_type, input_columns, request_data):
  if model_type == 'custom':
    data = preprocess_custom(request_data)
    prediction = model.predict(data)
  elif model_type == 'azureml':
    data = preprocess_azureml(request_data, input_columns)
    prediction = model.forecast(data)
  else:
      raise ValueError(f"Unknown model_type '{model_type}'.")
  
  # Convert prediction to a JSON-serializable format
  if isinstance(prediction, (np.ndarray, list)):
    prediction = prediction.tolist()
  elif isinstance(prediction, pd.DataFrame):
    prediction = prediction.to_dict(orient='records')
  return prediction