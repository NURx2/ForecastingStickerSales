import pytest
from datetime import date
from src.entities.data_entity import SalesData

def test_sales_data_creation():
    sales_data = SalesData(
        id=1,
        date=date(2023, 1, 1),
        country='US',
        store='Store1',
        product='Product1',
        num_sold=100.0
    )
    
    assert sales_data.id == 1
    assert sales_data.date == date(2023, 1, 1)
    assert sales_data.country == 'US'
    assert sales_data.store == 'Store1'
    assert sales_data.product == 'Product1'
    assert sales_data.num_sold == 100.0

def test_sales_data_from_dict():
    data_dict = {
        'id': 1,
        'date': '2023-01-01',
        'country': 'US',
        'store': 'Store1',
        'product': 'Product1',
        'num_sold': 100.0
    }
    
    sales_data = SalesData.from_dict(data_dict)
    
    assert sales_data.id == 1
    assert sales_data.date == date(2023, 1, 1)
    assert sales_data.country == 'US'
    assert sales_data.store == 'Store1'
    assert sales_data.product == 'Product1'
    assert sales_data.num_sold == 100.0

def test_sales_data_to_dict():
    sales_data = SalesData(
        id=1,
        date=date(2023, 1, 1),
        country='US',
        store='Store1',
        product='Product1',
        num_sold=100.0
    )
    
    data_dict = sales_data.to_dict()
    
    assert data_dict['id'] == 1
    assert data_dict['date'] == '2023-01-01'
    assert data_dict['country'] == 'US'
    assert data_dict['store'] == 'Store1'
    assert data_dict['product'] == 'Product1'
    assert data_dict['num_sold'] == 100.0

def test_sales_data_invalid_date():
    data_dict = {
        'id': 1,
        'date': 'invalid-date',
        'country': 'US',
        'store': 'Store1',
        'product': 'Product1',
        'num_sold': 100.0
    }
    
    with pytest.raises(ValueError):
        SalesData.from_dict(data_dict)

def test_sales_data_invalid_num_sold():
    data_dict = {
        'id': 1,
        'date': '2023-01-01',
        'country': 'US',
        'store': 'Store1',
        'product': 'Product1',
        'num_sold': 'invalid'
    }
    
    with pytest.raises(ValueError):
        SalesData.from_dict(data_dict) 