
import argparse

import pytest
import pandas as pd
import torch
from torch.nn import RNN, LSTM, GRU, L1Loss, Linear

from train import TemperaturePredictor, BaseRNNModel, load_hyperparams, prepare_data_module, train
from data_module import TemperatureDataModule
from utils import get_project_root

@pytest.fixture(scope='function', name='data_module')
def data_module_fixture():
  df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=200, freq='10min'),
    'feature1': range(200),
    'feature2': range(200, 400),
    'feature3': range(400, 600),
    'feature4': range(600, 800),
    'feature5': range(800, 1000),
    'feature6': range(1000, 1200),
    'feature7': range(1200, 1400),
    'feature8': range(1400, 1600),
    'feature9': range(1600, 1800),
    'feature10': range(1800, 2000),
    'feature11': range(2000, 2200),
    'feature12': range(2200, 2400),
    'feature13': range(2400, 2600),
    'T': range(2600, 2800)
  })
  return TemperatureDataModule(df, batch_size=8)

@pytest.fixture(scope='function', name='model')
def model_fixture(data_module):
  data_module.setup('fit')
  input_size = data_module.train_dataset.features.shape[1]
  return BaseRNNModel(input_size=input_size, h=1)

class TestBaseRNNModel:
  def test_initialization(self, data_module):
    data_module.setup('fit')
    input_size = data_module.train_dataset.features.shape[1]
    model = BaseRNNModel(input_size=input_size, h=1)

    assert isinstance(model.rnn, RNN)
    assert isinstance(model.out, Linear)

    model = BaseRNNModel(input_size=input_size, h=1, model='gru')
    assert isinstance(model.rnn, GRU)

    model = BaseRNNModel(input_size=input_size, h=1, model='lstm')
    assert isinstance(model.rnn, LSTM)

  def test_forward(self, data_module):
    data_module.setup('fit')
    input_size = data_module.train_dataset.features.shape[1]
    model = BaseRNNModel(input_size=input_size, h=1, model='lstm')

    x = torch.randn(2, 4, input_size) # batch_size=2, w=4
    output = model(x)

    assert output.shape == (2, 1)

class TestTemperaturePredictor:
  def test_initialization(self, model):
    predictor = TemperaturePredictor(model=model)
    assert predictor.model == model
    assert isinstance(predictor.criterion, L1Loss)
    assert predictor.optimizer is torch.optim.Adam
    assert predictor.learning_rate == 1e-3

  def test_forward(self, model):
    predictor = TemperaturePredictor(model=model)
    x = torch.randn(2, 4, model.rnn.input_size) # batch_size=2, w=4
    output = predictor(x)

    assert output.shape == (2, 1)

  def test_process_step(self, model):
    predictor = TemperaturePredictor(model=model)
    x = torch.randn(2, 4, model.rnn.input_size) # batch_size=2, w=4
    y = torch.randn(2, 1) # batch_size=2, h=1

    loss = predictor.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

  def test_configure_optimizers(self, model):
    predictor = TemperaturePredictor(model=model)
    optimizers = predictor.configure_optimizers()
    assert isinstance(optimizers, torch.optim.Adam)

def test_load_hyperparams():
  hyperparams = load_hyperparams(args_list=[])
  assert hasattr(hyperparams, 'batch_size')
  assert hasattr(hyperparams, 'w')
  assert hasattr(hyperparams, 'h')
  assert hasattr(hyperparams, 'lr')
  assert hasattr(hyperparams, 'model')
  assert hasattr(hyperparams, 'epochs')
  assert hasattr(hyperparams, 'dropout')
  assert hasattr(hyperparams, 'hidden_size')
  assert hasattr(hyperparams, 'num_layers')
  assert hasattr(hyperparams, 'seed')
  assert hasattr(hyperparams, 'pooling')

@pytest.mark.skip(reason="Requires internet connection to download dataset from Kaggle")
def test_prepare_data_module():
  data_module = prepare_data_module(batch_size=32, w=4, h=1)
  assert isinstance(data_module, TemperatureDataModule)
  assert data_module.batch_size == 32
  assert data_module.w == 4
  assert data_module.h == 1

def test_train_loop(data_module):
  hparams = argparse.Namespace(
    batch_size=8,
    w=4,
    h=1,
    lr=1e-3,
    model='rnn',
    hidden_size=16,
    num_layers=1,
    dropout=0.0,
    pooling='last',
    epochs=1
  )

  train(
    data_module=data_module,
    hparams=hparams,
    plot=False,
    logger=False
  )

  chk_path = get_project_root() / 'models' / 'rnn.ckpt'
  assert chk_path.exists()

  chk_path.unlink()
