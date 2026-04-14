
from pathlib import Path

import yaml

def get_project_root() -> Path:
  return Path(__file__).resolve().parents[1]

def load_config(config_name: str) -> dict:
  config_path = get_project_root() / 'config' / f'{config_name}.yaml'

  with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
  return config
