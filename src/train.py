
import argparse
from pathlib import Path

import torch
from torch import nn
from torchmetrics import MeanSquaredError
from pytorch_lightning import seed_everything, LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

from data_module import TemperatureDataModule
from utils import load_config

config = load_config('hyperparams')
SEED = config['training_config']['seed']
DEFAULT_BATCH_SIZE = config['training_config']['batch_size']
DEFAULT_W = config['training_config']['w']
DEFAULT_H = config['training_config']['h']
DEFAULT_LR = config['training_config']['lr']
DEFAULT_MODEL = config['training_config']['model']
seed_everything(SEED)

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

def main():
  # Hiperparametros
  parser = argparse.ArgumentParser(description='Train a temperature predictor model.')
  parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
  parser.add_argument('--w', type=int, default=DEFAULT_W, help='Window size for input sequences')
  parser.add_argument('--h', type=int, default=DEFAULT_H, help='Horizon for prediction')
  parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate for the optimizer')
  parser.add_argument(
    '--model', 
    type=str,
    default=DEFAULT_MODEL,
    choices=['rnn', 'lstm', 'gru'],
    help='Type of RNN model to use'
  )

  args = parser.parse_args()
  batch_size = args.batch_size
  w = args.w
  h = args.h
  lr = args.lr
  model_name = args.model

  df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'alistairking/weather-long-term-time-series-forecasting',
    'cleaned_weather.csv',
    pandas_kwargs={'parse_dates': ['date']}
  )

  data_module = TemperatureDataModule(df, batch_size=batch_size, w=w, h=h)
  data_module.setup('fit')
  input_size = data_module.train_dataset.features.shape[1]
  chk_path = Path('models')

  model = BaseRNNModel(input_size=input_size, h=h, model=model_name)
  module = TemperaturePredictor(model, learning_rate=lr, optimizer=torch.optim.Adam)

  trainer = Trainer(
    deterministic=True,
    callbacks = [
      EarlyStopping(monitor='val_loss', patience=5),
      ModelCheckpoint(monitor='val_loss', filename=model_name, dirpath=chk_path, enable_version_counter=False),
      PlotCallback()
    ],
    max_epochs=50
  )

  trainer.fit(module, data_module)
  trainer.test(module, data_module)

if __name__ == "__main__":
  main()
