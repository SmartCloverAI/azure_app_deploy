from fastapi import FastAPI, Request
import uvicorn
import json
import pickle
import os

from serving.scorers import score

app = FastAPI()

models_servers = {}


def init_models():
  # Load endpoint configurations from a JSON file
  with open('endpoints.json', 'r') as f:
    endpoint_configs = json.load(f)
  # Dynamically create endpoints based on the JSON configuration
  for endpoint in endpoint_configs:
    path = endpoint['path']
    model_location = endpoint['model_location']
    model_type = endpoint['model_type']
    input_columns = endpoint.get('input_columns', None)
    
    # Load the model from the specified location
    model_path = os.path.join(os.path.dirname(__file__), model_location)
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
      

def create_endpoint_func(model, model_type, input_columns):
  async def endpoint_func(request: Request):
    request_data = await request.json()
    try:
      prediction = score(model, model_type, input_columns, request_data)
      return {"prediction": prediction}
    except Exception as e:
      return {"error": str(e)}
  return endpoint_func



for model_server in model_servers:
  
  # Create and register the endpoint function
  endpoint_func = create_endpoint_func(model, model_type, input_columns)
  app.post(path)(endpoint_func)

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run("your_module_name:app", host="0.0.0.0", port=8000)
