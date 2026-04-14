
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

SEED = 42

class TemperatureDataset(Dataset):
  def __init__(self, df, w=4, h=1):
    self.features = df.drop(columns=['date', 'T']).values
    self.target = df['T'].values
    self.w = w
    self.h = h

  def __len__(self):
    return len(self.features) - (self.w + self.h) + 1

  def __getitem__(self, idx):
    features = self.features[idx:idx + self.w]
    target = self.target[idx + self.w: idx + self.w + self.h]
    return features, target
  
class TemperatureDataModule(LightningDataModule):
  def __init__(self, df, w=4, h=1, batch_size=16, val_size=0.1, test_size=0.2, reduction_strategy=None):
    super().__init__()
    # Inicalizamos los atributos de la clase
    self.data = df
    self.w = w
    self.h = h
    self.batch_size = batch_size

    # Preprocesamos el dataset
    self.data.drop_duplicates(inplace=True) # Quitamos duplicados
    self.impute_missing_values() # Imputamos valores faltantes
    self.train_df, self.val_df, self.test_df = self.sequential_train_val_test_split(val_size=val_size, test_size=test_size) # Split
    self.feature_scaler, self.target_scaler = self.normalize() # Normalizamos
    self.reductor = self.feature_reduction(reduction_strategy) # Reducimos dimensionalidad

  def setup(self, stage=None):
    if stage == 'fit':
      self.train_dataset = TemperatureDataset(self.train_df, w=self.w, h=self.h)
      self.val_dataset = TemperatureDataset(self.val_df, w=self.w, h=self.h)
    elif stage == 'test':
      self.test_dataset = TemperatureDataset(self.test_df, w=self.w, h=self.h)

  def impute_missing_values(self):
    # Buscamos fechas faltante
    freq = '10min' # Frecuencia de 10 minutors
    date_range = pd.date_range(start=self.data.date.min(), end=self.data.date.max(), freq=freq)
    missing_dates = date_range[~date_range.isin(self.data.date)]

    # Agregamos las fechas
    missing_df = pd.DataFrame({ 'date': missing_dates })
    self.data = pd.concat([self.data, pd.DataFrame({ 'date': missing_dates })]).reset_index(drop=True)

    # Interpolamos los datos faltantes
    self.data.set_index('date', inplace=True)
    self.data.interpolate(method='time', inplace=True)
    self.data.reset_index(inplace=True)

  def sequential_train_val_test_split(self, val_size, test_size):
    self.data.sort_values('date', inplace=True)

    # Calculamos los ínidices para hacer los splits
    n = len(self.data)
    train_end = int((1 - val_size - test_size) * n)
    val_end = int((1 - test_size) * n)

    # Hacemos el split
    train = self.data.iloc[:train_end].copy()
    val = self.data.iloc[train_end:val_end].copy()
    test = self.data.iloc[val_end:].copy()

    return train, val, test

  def normalize(self):
    # Inicializamos el scaler
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Seleccionamos las features a transformar
    numerical_features = self.data.select_dtypes(include='number').columns.tolist()
    numerical_features.remove('T')
    target = ['T']

    # Ajustamos con train y transformamos todo
    self.train_df.loc[:, numerical_features] = feature_scaler.fit_transform(self.train_df[numerical_features])
    self.train_df.loc[:, target] = target_scaler.fit_transform(self.train_df[target])

    self.val_df.loc[:, numerical_features] = feature_scaler.transform(self.val_df[numerical_features])
    self.val_df.loc[:, target] = target_scaler.transform(self.val_df[target])
    self.test_df.loc[:, numerical_features] = feature_scaler.transform(self.test_df[numerical_features])
    self.test_df.loc[:, target] = target_scaler.transform(self.test_df[target])
    return feature_scaler, target_scaler

  def feature_reduction(self, strategy, k=12):
    # Seleccionamos las features que pueden ser modificadas
    features_to_transform = self.data.select_dtypes(include='number').columns.tolist()
    features_to_transform.remove('T')

    if strategy == 'pca':
      # Inicializamos PCA
      pca = PCA(n_components=k, random_state=SEED)

      # Ajustamos con train y transformamos todo
      pca_train = pca.fit_transform(self.train_df[features_to_transform])
      pca_val = pca.transform(self.val_df[features_to_transform])
      pca_test = pca.transform(self.test_df[features_to_transform])

      # Ajustamos datos
      for i in range(pca_train.shape[1]):
        self.train_df[f'pca_{i+1}'] = pca_train[:, i]
        self.val_df[f'pca_{i+1}'] = pca_val[:, i]
        self.test_df[f'pca_{i+1}'] = pca_test[:, i]

      # Eliminamos datos originales
      self.train_df.drop(columns=features_to_transform, inplace=True)
      self.val_df.drop(columns=features_to_transform, inplace=True)
      self.test_df.drop(columns=features_to_transform, inplace=True)

      return pca

    elif strategy == 'selection':
      # Inicializamos KBest
      k_best = SelectKBest(mutual_info_regression, k=k)

      # Ajustamos con train
      k_best.fit(self.train_df[features_to_transform], self.train_df['T'])

      # Features finales
      best_features = ['date', 'T'] + k_best.get_feature_names_out().tolist()

      # Filtramos
      self.train_df = self.train_df[best_features]
      self.val_df = self.val_df[best_features]
      self.test_df = self.test_df[best_features]

      return k_best

  def collate_fn(self, batch):
    features, targets = zip(*batch)

    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)

    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return features, targets

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
