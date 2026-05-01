import wandb
import pickle
import torch
from pathlib import Path

from src.model_arch.BaseRNN import BaseRNNModel
from src.utils import load_config
import logging
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_model_and_preprocessor():
    logger.info("Loading model and preprocessor from W&B artifacts")
    config = load_config("hyperparams")
    model_cfg = config["training_config"]

    run = wandb.init(project="temperature-forecasting", job_type="inference")

    # --- MODEL ---
    model_artifact = run.use_artifact(
        "eisler-aguilar-universidad-polit-cnica-de-madrid/temperature-forecasting/lstm-clean:v0"
    )
    model_dir = model_artifact.download()
    model_pt = next(Path(model_dir).rglob("*.pt"), None)
    if model_pt is None:
        raise FileNotFoundError("No .pt found in model artifact")

    checkpoint = torch.load(model_pt, map_location="cpu", weights_only=True)

    input_size = (
        checkpoint.get("input_size")
        or model_artifact.metadata.get("input_size")
    )
    if input_size is None:
        raise ValueError("No se encontró input_size ni en el checkpoint ni en el metadata del artifact.")

    base_model = BaseRNNModel(
        input_size=input_size,
        h=model_cfg["h"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        pooling=model_cfg["pooling"],
        model=model_cfg["model_name"],
    )

    base_model.load_state_dict(checkpoint["state_dict"])
    base_model.eval()
    logger.info("Model loaded and set to inference mode")
    # --- PREPROCESSOR ---
    preproc_artifact = run.use_artifact(
        "eisler-aguilar-universidad-polit-cnica-de-madrid/temperature-forecasting/preprocessing-artifacts:v3"
    )
    preproc_dir = preproc_artifact.download()
    scaler_file = next(Path(preproc_dir).rglob("*.pkl"), None)
    if scaler_file is None:
        raise FileNotFoundError("No scaler found in preprocessing artifact")

    with open(scaler_file, "rb") as f:
        preprocessor = pickle.load(f)

    logger.info("Preprocessor loaded")
    run.finish()
    return base_model, preprocessor