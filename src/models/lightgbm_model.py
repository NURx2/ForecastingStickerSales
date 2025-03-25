import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List
from .base_model import BaseModel
from ..utils.logger import setup_logger
from sklearn.metrics import mean_absolute_percentage_error

class LightGBMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feature_cols = (
            config['features']['time_features'] +
            config['features']['categorical_features']
        )
        self.logger = setup_logger('lightgbm_model')
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        self.logger.info("Preparing LightGBM datasets")
        try:
            train_dataset = lgb.Dataset(
                train_data[self.feature_cols],
                label=train_data[self.config['data']['target_column']]
            )
            val_dataset = lgb.Dataset(
                val_data[self.feature_cols],
                label=val_data[self.config['data']['target_column']],
                reference=train_dataset
            )
            
            self.logger.info("Starting model training")
            self.model = lgb.train(
                params=self.config['model']['params'],
                train_set=train_dataset,
                num_boost_round=self.config['model']['params']['n_estimators'],
                valid_sets=[train_dataset, val_dataset],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.config['model']['params']['early_stopping_rounds']),
                    lgb.log_evaluation(period=100)
                ]
            )
            self.logger.info(f"Model training completed. Best iteration: {self.model.best_iteration}")
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}")
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.logger.error("Model has not been trained yet")
            raise RuntimeError("Model has not been trained yet.")
        
        self.logger.info(f"Generating predictions for {len(data)} samples")
        try:
            predictions = self.model.predict(data[self.feature_cols])
            result = pd.DataFrame({
                'id': data['id'],
                'num_sold': predictions
            })
            self.logger.info("Predictions generated successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Failed to generate predictions: {str(e)}")
    
    def evaluate(self, data: pd.DataFrame) -> float:
        if self.model is None:
            self.logger.error("Model has not been trained yet")
            raise RuntimeError("Model has not been trained yet.")
        
        self.logger.info("Evaluating model performance")
        try:
            predictions = self.model.predict(data[self.feature_cols])
            mape = mean_absolute_percentage_error(
                data[self.config['data']['target_column']],
                predictions
            ) * 100
            self.logger.info(f"Model MAPE: {mape:.2f}%")
            return mape
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")
    
    def save(self, path: str) -> None:
        if self.model is None:
            self.logger.error("No model to save")
            raise RuntimeError("No model to save.")
        
        self.logger.info(f"Saving model to {path}")
        try:
            self.model.save_model(path)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self, path: str) -> None:
        self.logger.info(f"Loading model from {path}")
        try:
            self.model = lgb.Booster(model_file=path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}") 