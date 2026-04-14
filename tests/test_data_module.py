
import pytest
import pandas as pd

from data_module import TemperatureDataModule, TemperatureDataset

@pytest.fixture
def sample_dataframe():
  data = {
    'date': pd.date_range(start='2023-01-01', periods=10, freq='10min'),
    'feature1': range(10),
    'feature2': range(10, 20),
    'T': range(20, 30)
  }
  return pd.DataFrame(data)

class TestTemperatureDataset:
  def test_number_of_samples(self, sample_dataframe):
    # Inicializamos el dataset
    dataset = TemperatureDataset(sample_dataframe, w=4, h=1)

    assert len(dataset) == 6

  def test_getitem(self, sample_dataframe):
    dataset = TemperatureDataset(sample_dataframe, w=4, h=1)

    assert len(dataset[0]) == 2

    features, target = dataset[0]
    assert features.shape == (4, 2)
    assert target.shape == (1,)

  def test_getitem_values(self, sample_dataframe):
    dataset = TemperatureDataset(sample_dataframe, w=4, h=1)

    features, target = dataset[0]
    assert (features == [[0, 10], [1, 11], [2, 12], [3, 13]]).all()
    assert (target == [24]).all()
