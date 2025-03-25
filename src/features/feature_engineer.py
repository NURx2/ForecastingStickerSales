import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from ..utils.logger import setup_logger

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_features = config['features']['time_features']
        self.categorical_features = config['features']['categorical_features']
        self.logger = setup_logger('feature_engineer')
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Creating time-based features")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        if 'year' in self.time_features:
            df['year'] = df['date'].dt.year
        if 'month' in self.time_features:
            df['month'] = df['date'].dt.month
        if 'day' in self.time_features:
            df['day'] = df['date'].dt.day
        if 'dayofweek' in self.time_features:
            df['dayofweek'] = df['date'].dt.dayofweek
        if 'quarter' in self.time_features:
            df['quarter'] = df['date'].dt.quarter
        if 'is_weekend' in self.time_features:
            df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        if 'is_holiday' in self.time_features:
            # Simple holiday detection (weekends and major western holidays)
            df['is_holiday'] = (
                (df['date'].dt.dayofweek.isin([5, 6])) |  # Weekends
                ((df['date'].dt.month == 1) & (df['date'].dt.day == 1)) |  # New Year's Day
                ((df['date'].dt.month == 12) & (df['date'].dt.day == 25))  # Christmas
            ).astype(int)
        
        self.logger.info(f"Created time features: {', '.join(self.time_features)}")
        return df
    
    def handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Converting categorical features")
        df = df.copy()
        for feature in self.categorical_features:
            df[feature] = df[feature].astype('category')
        self.logger.info(f"Converted categorical features: {', '.join(self.categorical_features)}")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting feature engineering pipeline")
        df = self.create_time_features(df)
        df = self.handle_categorical_features(df)
        self.logger.info("Feature engineering completed")
        return df
    
    def get_feature_columns(self) -> List[str]:
        return self.time_features + self.categorical_features 