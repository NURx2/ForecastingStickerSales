from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> float:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass 