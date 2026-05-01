from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import numpy as np

from src.inference.loader import load_model_and_preprocessor
from src.data.preprocessing import Preprocessor
from src.utils import load_config

import logging
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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
async def lifespan(app: FastAPI):
    model, scaler = load_model_and_preprocessor()
    config = load_config("hyperparams")

    logger.info("Model and preprocessor loaded successfully, setting up application state...")
    app.state.model = model
    app.state.preprocessor = Preprocessor(scaler)
    app.state.w = config["training_config"]["w"]

    logger.info("Application state set up complete, API is ready to serve requests.")

    yield
    logger.info("Shutting down and cleaning up...")


app = FastAPI(lifespan=lifespan)  


@app.post("/predict")
def predict(data: InputData):
    raw = list(data.dict().values())

    X = app.state.preprocessor.transform(raw)

    window = np.zeros((1, app.state.w, X.shape[1]), dtype=np.float32)
    window[0, -1, :] = X[0]

    X_tensor = torch.tensor(window)

    with torch.no_grad():
        pred = app.state.model(X_tensor)
    logger.info(f"Prediction made successfully: {pred.item()}")
    return OutputData(prediction=pred.item())
    