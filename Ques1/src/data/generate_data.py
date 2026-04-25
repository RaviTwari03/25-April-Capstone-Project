import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict
import os

class RetailDataGenerator:
    """Generate synthetic retail demand data for demonstration purposes."""
    
    def __init__(self, num_products: int = 100, num_regions: int = 10):
        self.num_products = num_products
        self.num_regions = num_regions
        self.product_categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports', 'Books']
        self.regions = [f'Region_{i}' for i in range(1, num_regions + 1)]
        
    def generate_products(self) -> pd.DataFrame:
        """Generate product information."""
        products = []
        for i in range(1, self.num_products + 1):
            products.append({
                'product_id': f'PROD_{i:04d}',
                'category': random.choice(self.product_categories),
                'base_price': round(random.uniform(10, 500), 2),
                'seasonal_factor': random.uniform(0.8, 1.2)
            })
        return pd.DataFrame(products)
    
    def generate_demand_data(self, start_date: str = '2022-01-01', 
                           end_date: str = '2023-12-31') -> pd.DataFrame:
        """Generate demand data with seasonal patterns and trends."""
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate products
        products_df = self.generate_products()
        
        # Generate demand data
        demand_records = []
        
        for date in dates:
            # Add seasonal effects
            month = date.month
            day_of_week = date.dayofweek
            is_weekend = day_of_week >= 5
            is_holiday = self._is_holiday(date)
            
            # Seasonal multiplier
            seasonal_multiplier = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier *= 1.5
            elif month in [6, 7, 8]:  # Summer
                seasonal_multiplier *= 1.2
            elif month in [1, 2]:  # Winter
                seasonal_multiplier *= 0.8
                
            for _, product in products_df.iterrows():
                for region in self.regions:
                    # Base demand
                    base_demand = random.uniform(10, 100)
                    
                    # Apply factors
                    category_factor = self._get_category_factor(product['category'], month)
                    region_factor = random.uniform(0.7, 1.3)
                    price_factor = 1.0 / (1 + product['base_price'] / 100)
                    
                    # Calculate final demand
                    demand = (base_demand * 
                             seasonal_multiplier * 
                             product['seasonal_factor'] * 
                             category_factor * 
                             region_factor * 
                             price_factor)
                    
                    # Add weekend and holiday effects
                    if is_weekend:
                        demand *= 1.3
                    if is_holiday:
                        demand *= 1.8
                        
                    # Add some noise
                    demand += random.gauss(0, demand * 0.1)
                    demand = max(0, round(demand, 2))
                    
                    demand_records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'product_id': product['product_id'],
                        'region': region,
                        'demand': demand,
                        'price': product['base_price'] * random.uniform(0.9, 1.1),
                        'category': product['category'],
                        'is_weekend': is_weekend,
                        'is_holiday': is_holiday,
                        'month': month,
                        'day_of_week': day_of_week
                    })
        
        return pd.DataFrame(demand_records)
    
    def _get_category_factor(self, category: str, month: int) -> float:
        """Get seasonal factor for product category."""
        category_seasonal = {
            'Electronics': {11: 1.5, 12: 1.8, 1: 0.7, 2: 0.8},
            'Clothing': {3: 1.3, 4: 1.4, 9: 1.2, 10: 1.3, 11: 1.1, 12: 1.2},
            'Food': {11: 1.3, 12: 1.5, 6: 1.1, 7: 1.1, 8: 1.0},
            'Home': {3: 1.2, 4: 1.3, 5: 1.1, 9: 1.1, 10: 1.2},
            'Sports': {5: 1.3, 6: 1.4, 7: 1.5, 8: 1.4},
            'Books': {8: 1.2, 9: 1.3, 12: 1.4}
        }
        
        return category_seasonal.get(category, {}).get(month, 1.0)
    
    def _is_holiday(self, date) -> bool:
        """Simple holiday detection."""
        month, day = date.month, date.day
        
        # Major holidays
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 24), # Thanksgiving (approximate)
            (2, 14),  # Valentine's Day
            (10, 31), # Halloween
        ]
        
        return (month, day) in holidays
    
    def save_data(self, output_dir: str = 'data/raw'):
        """Generate and save the dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate demand data
        demand_df = self.generate_demand_data()
        
        # Split into train and test sets
        train_df = demand_df[demand_df['date'] < '2023-07-01']
        test_df = demand_df[demand_df['date'] >= '2023-07-01']
        
        # Save files
        demand_df.to_csv(f'{output_dir}/retail_demand_full.csv', index=False)
        train_df.to_csv(f'{output_dir}/retail_demand_train.csv', index=False)
        test_df.to_csv(f'{output_dir}/retail_demand_test.csv', index=False)
        
        # Save products info
        products_df = self.generate_products()
        products_df.to_csv(f'{output_dir}/products.csv', index=False)
        
        print(f"Data saved to {output_dir}/")
        print(f"Total records: {len(demand_df)}")
        print(f"Training records: {len(train_df)}")
        print(f"Test records: {len(test_df)}")
        
        return demand_df, train_df, test_df, products_df

if __name__ == "__main__":
    generator = RetailDataGenerator(num_products=50, num_regions=5)
    generator.save_data()
