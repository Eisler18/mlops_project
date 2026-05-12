
import pytest
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

from train.data_module import TemperatureDataModule, TemperatureDataset

@pytest.fixture(name='csv_filename', scope='function')
def sample_data_file(tmp_path, monkeypatch):
  data_dir = tmp_path / 'data'
  data_dir.mkdir()

  df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=20, freq='10min'),
    'feature1': range(20),
    'feature2': range(20, 40),
    'feature3': range(40, 60),
    'feature4': range(60, 80),
    'feature5': range(80, 100),
    'feature6': range(100, 120),
    'feature7': range(120, 140),
    'feature8': range(140, 160),
    'feature9': range(160, 180),
    'feature10': range(180, 200),
    'feature11': range(200, 220),
    'feature12': range(220, 240),
    'feature13': range(240, 260),
    'T': range(260, 280)
  })
  csv_filename = 'test_data.csv'
  csv_path = data_dir / csv_filename
  df.to_csv(csv_path, index=False)

  monkeypatch.setattr('data_module.get_project_root', lambda: tmp_path)

  return csv_filename

@pytest.fixture(name='df', scope='function')
def sample_dataframe(csv_filename, tmp_path):
  csv_path = tmp_path / 'data' / csv_filename
  return pd.read_csv(csv_path, parse_dates=['date'])

class TestTemperatureDataset:
  def test_number_of_samples(self, df):
    dataset = TemperatureDataset(df, w=4, h=1)

    assert len(dataset) == 16

  def test_getitem(self, df):
    dataset = TemperatureDataset(df, w=4, h=1)

    assert len(dataset[0]) == 2

    features, target = dataset[0]
    assert features.shape == (4, 13)
    assert target.shape == (1,)

class TestTemperatureDataModule:
  def test_initialization(self, csv_filename):
    data_module = TemperatureDataModule(csv_filename, w=4, h=1, batch_size=16, val_size=0.1, test_size=0.2)

    assert data_module.w == 4
    assert data_module.h == 1
    assert data_module.batch_size == 16
    assert isinstance(data_module.train_df, pd.DataFrame)
    assert isinstance(data_module.val_df, pd.DataFrame)
    assert isinstance(data_module.test_df, pd.DataFrame)
    assert len(data_module.train_df) == 14
    assert len(data_module.val_df) == 2
    assert len(data_module.test_df) == 4
    assert isinstance(data_module.feature_scaler, StandardScaler)
    assert isinstance(data_module.target_scaler, StandardScaler)

  def test_setup(self, csv_filename):
    data_module = TemperatureDataModule(csv_filename, w=4, h=1, batch_size=16, val_size=0.1, test_size=0.2)
    data_module.setup(stage='fit')

    assert hasattr(data_module, 'train_dataset')
    assert hasattr(data_module, 'val_dataset')
    assert isinstance(data_module.train_dataset, TemperatureDataset)
    assert isinstance(data_module.val_dataset, TemperatureDataset)
    assert not hasattr(data_module, 'test_dataset')

    data_module.setup(stage='test')
    assert hasattr(data_module, 'test_dataset')
    assert isinstance(data_module.test_dataset, TemperatureDataset)

  def test_reductor(self, csv_filename):
    data_module = TemperatureDataModule(
      csv_filename, w=4, h=1, batch_size=16, val_size=0.1, test_size=0.2, reduction_strategy='pca'
    )
    assert isinstance(data_module.reductor, PCA)
    assert data_module.train_df.shape[1] == 14

    data_module = TemperatureDataModule(
      csv_filename, w=4, h=1, batch_size=16, val_size=0.1, test_size=0.2, reduction_strategy='selection'
    )
    assert isinstance(data_module.reductor, SelectKBest)
    assert data_module.train_df.shape[1] == 14

    data_module = TemperatureDataModule(csv_filename, w=4, h=1, batch_size=16, val_size=0.1, test_size=0.2)
    assert data_module.reductor is None
    assert data_module.train_df.shape[1] == 15
