"""
This module defines the FastAPI application and dynamically creates endpoints
based on the models loaded by ModelsHandler.
"""
import os

from fastapi import FastAPI, Request, HTTPException
from config import ModelsHandler

from utils import log_with_color, APP_NAME, ENV_VERSION


log_with_color(f"Starting FastAPI app {APP_NAME} in {ENV_VERSION} environment", color="b")

# Initialize ModelsHandler
MODEL_PATH = "./models/config.json"
models_handler = ModelsHandler(config_path=MODEL_PATH)

app = FastAPI()


# Dynamically create endpoints for each model
for model_name in models_handler.models.keys():
  # Create an endpoint function
  async def endpoint_func(request: Request, model_name=model_name):
    """
    A dynamic endpoint function that forwards the request data to the model's run function.

    Parameters
    ----------
    request : Request
      The incoming FastAPI request object.
    model_name : str
      The name of the model this endpoint corresponds to.

    Returns
    -------
    dict
      A dictionary with the model's inference results.
    """
    request_data = await request.body()
    request_data = str(request_data, "utf-8")
    try:
      model = models_handler.get_model(model_name)
    except KeyError:
      log_with_color(f"Failed to find model {model_name}", color="r")
      raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    try:
      result = model.run(request_data)
    except Exception as e:
      log_with_color(f"Failed to run model {model_name}", color="r")
      raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return result
  log_with_color(f"Creating endpoint for model {model_name}", color="b")
  # Register the endpoint with FastAPI
  str_endpoint = f"/{model_name}"
  app.post(str_endpoint)(endpoint_func)
  log_with_color(f"Created endpoint {str_endpoint}", color="g")
