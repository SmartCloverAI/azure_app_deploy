import numpy as np
import pandas as pd

from serving.preprocessing import preprocess_custom, preprocess_azureml

    # path = endpoint['path']
    # model_location = endpoint['model_location']
    # model_type = endpoint['model_type']
    # input_columns = endpoint.get('input_columns', None)
    
    # # Load the model from the specified location
    # model_path = os.path.join(os.path.dirname(__file__), model_location)
    # with open(model_path, 'rb') as f:
    #   model = pickle.load(f)    


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