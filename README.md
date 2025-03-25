# Sticker Sales Forecasting

A machine learning project for forecasting sticker sales using LightGBM.

## Project Structure

```
├── configs/           # Configuration files for training and prediction
├── data/             # Training and testing datasets
├── models/           # Trained models and predictions
├── notebooks/        # Jupyter notebooks for analysis
├── src/             # Source code
│   ├── data/        # Data processing scripts
│   ├── entities/    # Data classes
│   ├── features/    # Feature engineering
│   └── models/      # Model training and prediction
└── tests/           # Unit tests
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the project in development mode:
```bash
pip install -e .
```

## Usage

### Training

To train the model:
```bash
python -m src.models.train
```

### Prediction

To generate predictions:
```bash
python -m src.models.predict
```

## Data

The dataset contains sales data for Kaggle-branded stickers from different stores across various countries.

## Project Organization

- `configs/`: Contains YAML configuration files for training and prediction pipelines
- `src/`: Source code organized by functionality
- `tests/`: Unit tests for the project 