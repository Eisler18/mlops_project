from contextlib import asynccontextmanager
import logging
import numpy as np
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.loader import load_model_and_preprocessor
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)  # pylint: disable=no-member

class InputData(BaseModel):
  p: float
  Tpot: float
  Tdew: float
  rh: float
  VPmax: float
  VPact: float
  VPdef: float
  sh: float
  H2OC: float
  rho: float
  wv: float
  max_wv: float
  wd: float
  rain: float
  raining: float
  SWDR: float
  PAR: float
  max_PAR: float
  Tlog: float

class OutputData(BaseModel):
  prediction: float

@asynccontextmanager
async def lifespan(application: FastAPI):
  model, preprocessor, w = load_model_and_preprocessor()
  logger.info("Model and preprocessor loaded successfully, setting up application state...")
  application.state.model = model
  application.state.preprocessor = preprocessor
  application.state.w = w
  logger.info("Application state set up complete, API is ready to serve requests.")
  yield
  logger.info("Shutting down and cleaning up...")

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
  return { "status": "ok" }


@app.post("/predict")
def predict(data: InputData):
  raw = list(data.dict().values())

  x = app.state.preprocessor.transform(raw)

  window = np.zeros((1, app.state.w, x.shape[1]), dtype=np.float32)
  window[0, -1, :] = x[0]

  x_tensor = torch.tensor(window)

  with torch.no_grad():
    pred = app.state.model(x_tensor)
  logger.info(f"Prediction made successfully: {pred.item()}")
  return OutputData(prediction=pred.item())
