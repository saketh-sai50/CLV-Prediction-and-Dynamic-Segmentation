import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import great_expectations as gx
from src.utils_io import load_config, get_watermark, set_watermark

CONFIG = load_config()

def generate_synthetic_data(num_customers=500, num_transactions=20000):
    """Generates and appends realistic synthetic data if needed."""
    seed = CONFIG['seeds']['data_generation_seed']
    np.random.seed(seed)
    
    output_path = CONFIG['data']['raw_path']
    if os.path.exists(output_path):
        print("Raw data already exists. Skipping generation.")
        return

    print("Generating synthetic raw data...")
    customers = [f"C{1000 + i}" for i in range(num_customers)]
    data = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(num_transactions):
        customer_id = np.random.choice(customers)
        transaction_date = start_date + timedelta(
            days=np.random.randint(0, 690), 
            hours=np.random.randint(0, 23)
        )
        # Add some quantity information
        quantity = np.random.randint(1, 10)
        unit_price = round(np.random.uniform(5.0, 150.0), 2)
        amount = round(quantity * unit_price, 2)
        
        data.append([customer_id, transaction_date, quantity, unit_price, amount])
    
    df = pd.DataFrame(data, columns=['CustomerID', 'TransactionDate', 'Quantity', 'UnitPrice', 'Amount'])
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to '{output_path}'")

def validate_data(df):
    """Validates the dataframe using Great Expectations."""
    print("Validating data...")
    ge_df = gx.from_pandas(df)
    suite_path = CONFIG['data']['great_expectations_suite']
    validation_result = ge_df.validate(expectation_suite=suite_path)
    if not validation_result["success"]:
        print("Data validation failed!")
        # In a real pipeline, you would raise an exception or send an alert
        # For this example, we'll just print the failed expectations
        for result in validation_result["results"]:
            if not result["success"]:
                print(result["expectation_config"]["kwargs"])
        raise ValueError("Data validation failed. Aborting.")
    print("Data validation successful.")

def load_new_data():
    """Loads, validates, and returns only new transactions since the last run."""
    generate_synthetic_data()
    
    watermark_path = CONFIG['data']['watermark_path']
    last_run_timestamp = get_watermark(watermark_path)
    
    print(f"Loading data since last run at: {last_run_timestamp}")
    df = pd.read_csv(CONFIG['data']['raw_path'], parse_dates=['TransactionDate'])
    
    new_data = df[df['TransactionDate'] > last_run_timestamp].copy()
    
    if new_data.empty:
        print("No new data to process.")
        return None
        
    validate_data(new_data)
    
    # Update watermark to the latest timestamp in the new data
    set_watermark(watermark_path, new_data['TransactionDate'].max())
    
    return new_data

if __name__ == "__main__":
    df_new = load_new_data()
    if df_new is not None:
        print(f"\nSuccessfully loaded and validated {len(df_new)} new records.")