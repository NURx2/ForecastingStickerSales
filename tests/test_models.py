import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.models.lightgbm_model import LightGBMModel
from src.models.model_factory import ModelFactory

@pytest.fixture
def model_config():
    return {
        'data': {
            'target_column': 'num_sold'
        },
        'features': {
            'time_features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'],
            'categorical_features': ['country', 'store', 'product']
        },
        'model': {
            'name': 'lightgbm',
            'params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 100,
                'early_stopping_rounds': 10
            }
        }
    }

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    countries = ['US', 'UK', 'CA']
    stores = ['Store1', 'Store2', 'Store3']
    products = ['Product1', 'Product2', 'Product3']
    
    data = []
    for date in dates:
        for country in countries:
            for store in stores:
                for product in products:
                    data.append({
                        'id': len(data) + 1,
                        'date': date,
                        'country': country,
                        'store': store,
                        'product': product,
                        'num_sold': np.random.randint(0, 100)
                    })
    
    return pd.DataFrame(data)

def test_train_model(model_config, sample_data):
    model = LightGBMModel(model_config)
    train_data = sample_data.iloc[:100]
    val_data = sample_data.iloc[100:200]
    
    model.train(train_data, val_data)
    assert isinstance(model.model, lgb.Booster)

def test_evaluate_model(model_config, sample_data):
    model = LightGBMModel(model_config)
    train_data = sample_data.iloc[:100]
    val_data = sample_data.iloc[100:200]
    
    model.train(train_data, val_data)
    rmse = model.evaluate(val_data)
    assert isinstance(rmse, float)
    assert rmse > 0

def test_generate_predictions(model_config, sample_data):
    model = LightGBMModel(model_config)
    train_data = sample_data.iloc[:100]
    val_data = sample_data.iloc[100:200]
    test_data = sample_data.iloc[200:300]
    
    model.train(train_data, val_data)
    predictions = model.predict(test_data)
    
    assert isinstance(predictions, pd.DataFrame)
    assert 'id' in predictions.columns
    assert 'num_sold' in predictions.columns
    assert len(predictions) == len(test_data)
    assert not predictions['num_sold'].isnull().any()

def test_save_load_model(model_config, sample_data, tmp_path):
    model = LightGBMModel(model_config)
    train_data = sample_data.iloc[:100]
    val_data = sample_data.iloc[100:200]
    
    model.train(train_data, val_data)
    model_path = tmp_path / "model.txt"
    model.save(str(model_path))
    
    loaded_model = LightGBMModel(model_config)
    loaded_model.load(str(model_path))
    
    assert isinstance(loaded_model.model, lgb.Booster)
    assert loaded_model.model.best_iteration == model.model.best_iteration

def test_model_factory(model_config):
    model = ModelFactory.create_model(model_config)
    assert isinstance(model, LightGBMModel)
    
    with pytest.raises(ValueError):
        model_config['model']['name'] = 'unknown_model'
        ModelFactory.create_model(model_config) 