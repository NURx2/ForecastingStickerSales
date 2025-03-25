import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
from ..features.feature_engineer import FeatureEngineer
from ..utils.logger import setup_logger

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.logger = setup_logger('data_processor')
    
    def load_data(self, prediction_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Loading data")
        try:
            if prediction_mode:
                # For prediction, we only need test data
                test_df = pd.read_csv(self.config['data']['test_path'])
                self.logger.info(f"Loaded {len(test_df)} test samples")
                return pd.DataFrame(), test_df  # Return empty DataFrame for train
            else:
                # For training, we need both train and test data
                train_df = pd.read_csv(self.config['data']['train_path'])
                test_df = pd.read_csv(self.config['data']['test_path'])
                self.logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
                return train_df, test_df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Creating time features")
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = self.feature_engineer.create_time_features(df)
            self.logger.info("Time features created successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise RuntimeError(f"Failed to create time features: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        self.logger.info("Preprocessing data")
        try:
            # Handle NaN values in target column
            if is_training and self.config['data']['target_column'] in df.columns:
                if df[self.config['data']['target_column']].isna().any():
                    self.logger.warning(f"Found {df[self.config['data']['target_column']].isna().sum()} NaN values in target column")
                    df = df.dropna(subset=[self.config['data']['target_column']])
            
            # Fill NaN values in features
            for col in self.feature_engineer.get_feature_columns():
                if df[col].isna().any():
                    self.logger.warning(f"Found {df[col].isna().sum()} NaN values in {col}")
                    if df[col].dtype == 'category':
                        df[col] = df[col].fillna('missing')
                    else:
                        df[col] = df[col].fillna(df[col].mean())
            
            df = self.feature_engineer.handle_categorical_features(df)
            self.logger.info("Data preprocessing completed")
            return df
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise RuntimeError(f"Failed to preprocess data: {str(e)}")
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Splitting data into train and validation sets")
        try:
            train_size = int(len(df) * (1 - self.config['training']['test_size']))
            train_df = df[:train_size]
            val_df = df[train_size:]
            self.logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation samples")
            return train_df, val_df
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise RuntimeError(f"Failed to split data: {str(e)}")
    
    def prepare_data(self, prediction_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.logger.info("Starting data preparation pipeline")
        try:
            train_df, test_df = self.load_data(prediction_mode=prediction_mode)
            
            if not prediction_mode:
                train_df = self.create_time_features(train_df)
            test_df = self.create_time_features(test_df)
            
            if not prediction_mode:
                train_df = self.preprocess_data(train_df, is_training=True)
            test_df = self.preprocess_data(test_df, is_training=False)
            
            if not prediction_mode:
                train_df, val_df = self.split_data(train_df)
            else:
                val_df = pd.DataFrame()  # Empty DataFrame for prediction mode
            
            self.logger.info("Data preparation pipeline completed")
            return train_df, val_df, test_df
        except Exception as e:
            self.logger.error(f"Error in data preparation pipeline: {str(e)}")
            raise RuntimeError(f"Failed to prepare data: {str(e)}") 