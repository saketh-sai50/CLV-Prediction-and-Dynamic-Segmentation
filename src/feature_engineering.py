import pandas as pd
from datetime import datetime
import os
from src.utils_io import load_config

CONFIG = load_config()

def calculate_rfm(df, snapshot_date):
    """Calculates Recency, Frequency, and Monetary features."""
    rfm = df.groupby('CustomerID').agg(
        Recency=('TransactionDate', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionDate', 'count'),
        MonetaryValue=('Amount', 'sum')
    ).reset_index()
    return rfm

def add_behavioral_features(df, snapshot_date):
    """Calculates additional behavioral features."""
    behavioral = df.groupby('CustomerID').agg(
        FirstPurchaseDate=('TransactionDate', 'min'),
        LastPurchaseDate=('TransactionDate', 'max'),
        AvgOrderValue=('Amount', 'mean'),
        SpendingVolatility=('Amount', 'std'),
        UniqueProducts=('UnitPrice', 'nunique') # Proxy for product diversity
    ).reset_index()

    behavioral['CustomerTenure'] = (snapshot_date - behavioral['FirstPurchaseDate']).dt.days
    
    # Interpurchase time
    purchase_dates = df.sort_values(by=['CustomerID', 'TransactionDate'])
    purchase_dates['InterpurchaseTime'] = purchase_dates.groupby('CustomerID')['TransactionDate'].diff().dt.days
    interpurchase = purchase_dates.groupby('CustomerID')['InterpurchaseTime'].mean().reset_index()
    interpurchase.rename(columns={'InterpurchaseTime': 'AvgInterpurchaseTime'}, inplace=True)

    # Merge all
    features = pd.merge(behavioral, interpurchase, on='CustomerID', how='left')
    return features.drop(columns=['FirstPurchaseDate', 'LastPurchaseDate'])

def build_feature_set():
    """Main function to build and save the complete feature set."""
    print("Building feature set...")
    raw_data_path = CONFIG['data']['raw_path']
    processed_feature_path = CONFIG['data']['processed_path']
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at {raw_data_path}. Run `make data` first.")

    df = pd.read_csv(raw_data_path, parse_dates=['TransactionDate'])
    
    # For CLV, we define a prediction period. Let's predict value in the last 90 days.
    # This is a common way to frame a supervised CLV problem.
    snapshot_date = df['TransactionDate'].max()
    train_end_date = snapshot_date - pd.Timedelta(days=90)
    
    # Features are built on data BEFORE the prediction period
    train_df = df[df['TransactionDate'] <= train_end_date]
    
    # Target (MonetaryValue) is calculated on data WITHIN the prediction period
    target_df = df[df['TransactionDate'] > train_end_date]
    clv_target = target_df.groupby('CustomerID')['Amount'].sum().reset_index()
    clv_target.rename(columns={'Amount': 'CLV_90_days'}, inplace=True)
    
    # Engineer features using training data
    rfm_features = calculate_rfm(train_df, train_end_date)
    behavioral_features = add_behavioral_features(train_df, train_end_date)
    
    # Combine all features
    features = pd.merge(rfm_features, behavioral_features, on='CustomerID', how='left')
    
    # Combine features with the target
    final_df = pd.merge(features, clv_target, on='CustomerID', how='left')
    
    # Fill NaNs - target may be NaN if customer didn't purchase in the last 90 days
    final_df['CLV_90_days'].fillna(0, inplace=True)
    final_df.fillna(0, inplace=True) # Fill other NaNs (e.g., for single-purchase customers)
    
    final_df.to_csv(processed_feature_path, index=False)
    print(f"Complete feature set saved to '{processed_feature_path}'")
    return final_df

if __name__ == "__main__":
    build_feature_set()