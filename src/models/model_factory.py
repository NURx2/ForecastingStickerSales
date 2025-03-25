from typing import Dict, Any
from .base_model import BaseModel
from .lightgbm_model import LightGBMModel

class ModelFactory:
    _models = {
        'lightgbm': LightGBMModel
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        model_name = config['model']['name'].lower()
        if model_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}")
        return cls._models[model_name](config) 