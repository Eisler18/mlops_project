import logging
import pickle
from pathlib import Path
import torch
import wandb
from src.logging.logging_config import setup_logging
from src.train import BaseRNNModel

setup_logging()
logger = logging.getLogger(__name__)  # pylint: disable=no-member


def load_model_and_preprocessor():
  logger.info("Loading model and preprocessor from W&B artifacts")

  run = wandb.init(project="temperature-forecasting", job_type="inference")
  try:
    # --- MODEL ---
    model_artifact = run.use_artifact(
        "eisler-aguilar-universidad-polit-cnica-de-madrid/temperature-forecasting/lstm-clean:v0"
    )
    model_dir = model_artifact.download()
    model_pt = next(Path(model_dir).rglob("*.pt"), None)
    if model_pt is None:
      raise FileNotFoundError("No .pt found in model artifact")

    training_run = model_artifact.logged_by()
    cfg = training_run.config
    logger.info("Hparams fetched from W&B run: %s", training_run.name)

    checkpoint = torch.load(model_pt, map_location="cpu", weights_only=True)
    base_model = BaseRNNModel(
        input_size=checkpoint["input_size"],
        h=cfg["h"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pooling=cfg["pooling"],
        model=cfg["model_name"],
    )
    base_model.load_state_dict(checkpoint["state_dict"])
    base_model.eval()
    logger.info("Model loaded and set to inference mode")
# --- PREPROCESSOR ---
    preproc_artifact = run.use_artifact(
        "eisler-aguilar-universidad-polit-cnica-de-madrid/temperature-forecasting/preprocessing-artifacts:v3"
    )
    preproc_dir = preproc_artifact.download()

    scaler_file = Path(preproc_dir) / "feature_scaler.pkl"
    if not scaler_file.exists():
        raise FileNotFoundError("feature_scaler.pkl not found in preprocessing artifact")

    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    from src.inference.preprocessing import Preprocessor
    preprocessor = Preprocessor(scaler)

    logger.info("Preprocessor loaded")
    return base_model, preprocessor
  finally:
    run.finish()
