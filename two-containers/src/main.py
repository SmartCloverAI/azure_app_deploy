
import os
import json

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from config import APP_NAME
from utils import log_with_color

from model import init, run


app = FastAPI()
model = None

  
@app.on_event("startup")
def load_model():
  env_ver = os.getenv("SERVING_ENV_VERSION")
  base_env_ver = os.getenv("BASE_SERVING_ENV_VERSION")
  log_with_color("Initiating startup process...", color="b")
  log_with_color(f"Base environment version: {base_env_ver}", color="b")
  log_with_color(f"Exec environment version: {env_ver}", color='b')
  init()
  log_with_color("Startup process done.", color="g")
  return


@app.post(f"/{APP_NAME}/")
async def predict(request: Request):
  """Endpoint to handle prediction requests."""
  try:
    log_with_color("Received predict request", color="b")
    request_json = await request.body()
    # Preprocess and predict
    log_with_color("Executing run on {}:\n{}".format(type(request_json), json.dumps(request_json)), color="b")
    result = run(request_json)
    return result
  except Exception as e:
    log_with_color(f"Error: {e} on request:\n{json.dumps(request_json, indent=2)}", color="r")
    raise HTTPException(status_code=500, detail=str(e))