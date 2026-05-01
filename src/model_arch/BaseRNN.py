import torch
from torch import nn
from pytorch_lightning import LightningModule, Callback
from torchmetrics import MeanSquaredError



import torch
from pathlib import Path


def export_ckpt(ckpt_path: str, output_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning usually stores weights here
    state_dict = ckpt.get("state_dict", ckpt)

    # Clean prefix if needed
    cleaned_state_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
    }

    torch.save(
        {"state_dict": cleaned_state_dict},
        output_path
    )

    print(f"Saved clean model to {output_path}")


if __name__ == "__main__":
    export_ckpt(
        "artifacts/lstm-v1/model.ckpt",
        "artifacts/lstm-v1/model.pt"
    )

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