from fastapi import FastAPI, Request
import uvicorn
import json
import pickle
import os

from serving.scorers import score

app = FastAPI()

model_servers = {}


def init_models():
  # Load endpoint configurations from a JSON file
  with open('endpoints.json', 'r') as f:
    endpoint_configs = json.load(f)
  # Dynamically create endpoints based on the JSON configuration
  for endpoint_config in endpoint_configs:
    model_name = endpoint['model_name']
    model_servers[model_name] = ModelServer(endpoint)
    
      

def create_endpoint_func(function):
  async def endpoint_func(request: Request):
    request_data = await request.json()
    try:
      prediction = function(request_data)
      return {"prediction": prediction}
    except Exception as e:
      return {"error": str(e)}
  return endpoint_func



for model_server in model_servers:  
  # Create and register the endpoint function
  path = model_server.get_url()
  function = model_server.get_function()
  endpoint_func = create_endpoint_func(function)
  app.post(path)(endpoint_func)

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run("your_module_name:app", host="0.0.0.0", port=8000)
