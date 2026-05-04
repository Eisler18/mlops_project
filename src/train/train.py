
import argparse
import uuid
from pathlib import Path

import torch
from torch import nn
from torchmetrics import MeanSquaredError
from pytorch_lightning import seed_everything, LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import wandb

from .data_module import TemperatureDataModule
from ..utils import load_config, get_project_root

# pylint: disable=arguments-differ
class TemperaturePredictor(LightningModule):
  def __init__(self, model, learning_rate=1e-3, optimizer=torch.optim.Adam):
    super().__init__()
    self.save_hyperparameters()
    self.learning_rate = learning_rate

    self.model = model
    self.optimizer = optimizer

    self.criterion = nn.L1Loss()
    self.rmse = MeanSquaredError(squared=False)

  def forward(self, x):
    return self.model(x)

  def process_step(self, batch, split='train'):
    inputs, targets = batch
    output = self(inputs)

    preds = output.view(-1)
    targets = targets.view(-1)

    loss = self.criterion(preds, targets)
    self.log_dict(
        {
            f'{split}_loss': loss,
            f'{split}_rmse': self.rmse(preds, targets),
        },
        on_epoch=True, on_step=False, prog_bar=True)

    return loss

  def training_step(self, batch, _batch_idx):
    return self.process_step(batch, 'train')

  def validation_step(self, batch, _batch_idx):
    return self.process_step(batch, 'val')

  def test_step(self, batch, _batch_idx):
    return self.process_step(batch, 'test')

  def configure_optimizers(self):
    return self.optimizer(self.parameters(), lr=self.learning_rate)

class PlotCallback(Callback):
  '''
  Grafica las pérdidas una vez el entrenamiento termina
  '''
  def __init__(self):
    super().__init__()
    self.losses = {}

  def on_train_epoch_end(self, trainer, _pl_module):
    # Guardamos los valores por epoca
    for key, value in trainer.callback_metrics.items():
      loss_name = key.replace('train_', '').replace('val_', '')
      if loss_name not in self.losses:
        self.losses[loss_name] = {}

      if key.startswith('train_'):
        self.losses[loss_name].setdefault('train', []).append(value.item())
      elif key.startswith('val_'):
        self.losses[loss_name].setdefault('val', []).append(value.item())

  def on_fit_end(self, _trainer, _pl_module):
    # Graficamos el historico del Loss
    num_losses = len(self.losses)
    num_rows = num_losses // 3 + 1 if num_losses % 3 != 0 else num_losses // 3
    num_columns = min(3, num_losses)

    _, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for (loss_name, losses), ax in zip(self.losses.items(), axes):
      ax.plot(losses['train'], label='Training')
      ax.plot(losses['val'], label='Validation')
      ax.set_xlabel('Epochs')
      ax.set_ylabel(loss_name.replace('_', ' ').title())
      ax.legend()

    plt.tight_layout()
    plt.show()

# pylint: disable=too-many-arguments
class BaseRNNModel(nn.Module):
  def __init__(self, input_size, *, h=1, hidden_size=64, num_layers=2, dropout=0.0, pooling='last', model='rnn'):
    super().__init__()
    # Posibles modelos
    models = { 'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU }

    # Inicializamos
    self.pooling = pooling
    self.rnn = models[model](
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout
    )
    self.out = nn.Linear(hidden_size, h)

  def forward(self, x):
    outputs, *_ = self.rnn(x)
    if self.pooling == 'last':
      x = outputs[:, -1, :]
    elif self.pooling == 'mean':
      x = torch.mean(outputs, dim=1)
    else:
      x = torch.amax(outputs, dim=1)
    x = self.out(x)
    return x
# pylint: enable=(arguments-differ, too-many-arguments)

def load_hyperparams(config_path='hyperparams', args_list=None):
  config = load_config(config_path)
  training_config = config['training_config']

  parser = argparse.ArgumentParser(description='Train a temperature predictor model.')
  parser.add_argument('--batch_size', type=int, default=training_config['batch_size'], help='Batch size for training')
  parser.add_argument('--w', type=int, default=training_config['w'], help='Window size for input sequences')
  parser.add_argument('--h', type=int, default=training_config['h'], help='Horizon for prediction')
  parser.add_argument('--lr', type=float, default=training_config['lr'], help='Learning rate for the optimizer')
  parser.add_argument(
    '--model_name', 
    type=str,
    default=training_config['model_name'],
    choices=['rnn', 'lstm', 'gru'],
    help='Type of RNN model to use'
  )
  parser.add_argument(
    '--dropout',
    type=float,
    default=training_config['dropout'],
    help='Dropout rate for the RNN layers'
  )
  parser.add_argument(
    '--hidden_size',
    type=int,
    default=training_config['hidden_size'],
    help='Number of hidden units in the RNN layers'
  )
  parser.add_argument('--num_layers', type=int, default=training_config['num_layers'], help='Number of RNN layers')
  parser.add_argument('--seed', type=int, default=training_config['seed'], help='Random seed for reproducibility')
  parser.add_argument(
    '--pooling',
    type=str,
    default=training_config['pooling'],
    choices=['last', 'mean', 'max'],
    help='Pooling strategy for RNN outputs'
  )
  parser.add_argument('--epochs', type=int, default=training_config['epochs'], help='Number of training epochs')
  parser.add_argument('--plot', action='store_true', help='Whether to plot training/validation losses after training')
  parser.add_argument(
    '--reduction_strategy',
    type=str,
    default=None,
    choices=['pca', 'selection', None],
    help='Feature reduction strategy (pca, selection, or None)'
  )

  return parser.parse_args(args_list)

def prepare_data_module(batch_size, w, h, reduction_strategy=None):
  df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'alistairking/weather-long-term-time-series-forecasting',
    'cleaned_weather.csv',
    pandas_kwargs={'parse_dates': ['date']}
  )

  return TemperatureDataModule(df, batch_size=batch_size, w=w, h=h, reduction_strategy=reduction_strategy)


def _export_model(trainer, module, hparams, input_size, wandb_logger):
  """Export trained model to .pt format and log to W&B."""
  best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
  pt_path = best_ckpt.with_suffix(".pt")

  torch.save(
      {
          "state_dict": module.model.state_dict(),
          "input_size": input_size,
          "model_name": hparams.model_name,
      },
      pt_path,
  )
  print(f"[export] Guardado {pt_path}")

  if wandb_logger:
      artifact = wandb.Artifact(
          name=f"{hparams.model_name}-clean",
          type="model",
          metadata={"input_size": input_size, "model_name": hparams.model_name},
      )
      artifact.add_file(str(pt_path))
      wandb_logger.experiment.log_artifact(artifact)
      wandb_logger.finalize("success")


# pylint: disable=too-many-arguments
def train(data_module, hparams, *, plot=True, logger=True):
  data_module.setup('fit')
  input_size = data_module.train_dataset.features.shape[1]
  chk_path = get_project_root() / 'models'

  model = BaseRNNModel(
      input_size=input_size,
      h=data_module.h,
      model=hparams.model_name,
      hidden_size=hparams.hidden_size,
      num_layers=hparams.num_layers,
      dropout=hparams.dropout,
      pooling=hparams.pooling
  )
  module = TemperaturePredictor(model, learning_rate=hparams.lr)

  if logger:
      config = {k: v for k, v in vars(hparams).items() if k != 'plot'}
      group_id = str(uuid.uuid4())
      preprocessing_artifact_ref = data_module.log_preprocessing_artifacts(group=group_id)
      wandb_logger = WandbLogger(
          project='temperature-forecasting',
          name=f'train_{group_id}',
          config={**config, 'preprocessing_artifact': preprocessing_artifact_ref},
          log_model=True,
          checkpoint_name=hparams.model_name,
          job_type='train',
          group=group_id
      )
      wandb_logger.use_artifact(preprocessing_artifact_ref)
  else:
      wandb_logger = None

  checkpoint_callback = ModelCheckpoint(
      monitor='val_loss',
      filename=hparams.model_name,
      dirpath=chk_path,
      enable_version_counter=False,
      save_top_k=1,
      mode='min'
  )

  callbacks = [
      EarlyStopping(monitor='val_loss', patience=5),
      checkpoint_callback
  ]
  if plot:
      callbacks.append(PlotCallback())

  trainer = Trainer(
      deterministic=True,
      callbacks=callbacks,
      max_epochs=hparams.epochs,
      logger=wandb_logger
  )

  trainer.fit(module, data_module)
  trainer.test(module, data_module)

  _export_model(trainer, module, hparams, input_size, wandb_logger)

if __name__ == "__main__":
  args = load_hyperparams()

  seed_everything(args.seed)

  train(
    data_module=prepare_data_module(args.batch_size, args.w, args.h, reduction_strategy=args.reduction_strategy),
    hparams=args,
    plot=args.plot
  )
