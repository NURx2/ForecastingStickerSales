import yaml
from pathlib import Path
from typing import Dict, Any
from .lightgbm_model import LightGBMModel
from ..data.data_processor import DataProcessor
from ..utils.logger import setup_logger

def load_config(config_path: str) -> Dict[str, Any]:
    logger = setup_logger('config_loader')
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        logger.info("Configuration loaded successfully")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise ValueError(f"Invalid YAML configuration: {str(e)}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise RuntimeError(f"Failed to load configuration: {str(e)}")

def predict(config: Dict[str, Any]) -> None:
    logger = setup_logger('predictor')
    logger.info("Starting prediction pipeline")
    
    try:
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info("Loading model")
        model = LightGBMModel(config)
        model.load(str(model_path))
        
        logger.info("Preparing data")
        data_processor = DataProcessor(config)
        _, _, test_df = data_processor.prepare_data(prediction_mode=True)
        
        logger.info("Generating predictions")
        predictions = model.predict(test_df)
        
        output_path = Path(config['output']['predictions_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Failed to generate predictions: {str(e)}")

def main():
    logger = setup_logger('main')
    logger.info("Starting prediction process")
    
    try:
        config = load_config("configs/predict_config.yaml")
        predict(config)
        logger.info("Prediction completed successfully")
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 