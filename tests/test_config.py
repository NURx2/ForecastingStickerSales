import pytest
import yaml
import os
from typing import Dict, Any

def test_load_train_config(tmp_path: str, sample_config: Dict[str, Any]) -> None:
    config_path = os.path.join(tmp_path, 'train_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    from src.models.train import load_config
    loaded_config = load_config(config_path)
    
    assert loaded_config['data']['target_column'] == 'num_sold'
    assert loaded_config['model']['name'] == 'lightgbm'
    assert 'params' in loaded_config['model']
    assert 'features' in loaded_config['data']

def test_load_predict_config(tmp_path: str, sample_config: Dict[str, Any]) -> None:
    config_path = os.path.join(tmp_path, 'predict_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    from src.models.predict import load_config
    loaded_config = load_config(config_path)
    
    assert loaded_config['data']['target_column'] == 'num_sold'
    assert loaded_config['model']['name'] == 'lightgbm'
    assert 'model_path' in loaded_config['model']
    assert 'output_path' in loaded_config['prediction']

def test_config_file_not_found() -> None:
    from src.models.train import load_config
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent_config.yaml') 