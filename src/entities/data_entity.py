from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class SalesData:
    id: int
    date: date
    country: str
    store: str
    product: str
    num_sold: float
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SalesData':
        return cls(
            id=data['id'],
            date=date.fromisoformat(data['date']),
            country=data['country'],
            store=data['store'],
            product=data['product'],
            num_sold=float(data['num_sold'])
        )
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'country': self.country,
            'store': self.store,
            'product': self.product,
            'num_sold': self.num_sold
        } 