import yaml
from pathlib import Path
from typing import Dict, Any
from .lightgbm_model import LightGBMModel
from ..data.data_processor import DataProcessor
from ..utils.logger import setup_logger

class ConfigError(Exception):
    pass

def load_config(config_path: str) -> Dict[str, Any]:
    logger = setup_logger('config_loader')
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ConfigError("Configuration must be a dictionary")
        logger.info("Configuration loaded successfully")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise ConfigError(f"Invalid YAML configuration: {str(e)}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise ConfigError(f"Configuration file not found: {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise RuntimeError(f"Failed to load configuration: {str(e)}")

def train_model(config: Dict[str, Any]) -> None:
    logger = setup_logger('trainer')
    logger.info("Starting model training pipeline")
    
    try:
        logger.info("Preparing data")
        data_processor = DataProcessor(config)
        train_df, val_df, test_df = data_processor.prepare_data()
        
        logger.info("Creating model")
        model = LightGBMModel(config)
        
        logger.info("Training model")
        model.train(train_df, val_df)
        
        logger.info("Evaluating model")
        mape = model.evaluate(val_df)
        logger.info(f"Validation MAPE: {mape:.2f}%")
        
        output_dir = Path(config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pkl"
        logger.info(f"Saving model to {model_path}")
        model.save(str(model_path))
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise RuntimeError(f"Failed to train model: {str(e)}")

def main():
    logger = setup_logger('main')
    logger.info("Starting training process")
    
    try:
        config = load_config("configs/train_config.yaml")
        train_model(config)
        logger.info("Training completed successfully")
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 