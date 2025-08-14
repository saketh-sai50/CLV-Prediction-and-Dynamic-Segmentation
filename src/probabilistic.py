import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import mlflow
import joblib
from src.utils_io import load_config

CONFIG = load_config()

def fit_probabilistic_models(df):
    """Fits BG/NBD and Gamma-Gamma models and returns features."""
    print("Fitting probabilistic models...")
    
    # lifetimes requires a specific format: frequency, recency, T, monetary_value
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col='CustomerID',
        datetime_col='TransactionDate',
        monetary_value_col='Amount'
    )
    
    # Fit BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    
    # Predict future purchases
    summary['predicted_purchases_90d'] = bgf.predict(
        t=90, 
        frequency=summary['frequency'], 
        recency=summary['recency'], 
        T=summary['T']
    )
    
    # Fit Gamma-Gamma model (only on customers who made repeat purchases)
    returning_customers_summary = summary[summary['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(
        returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value']
    )
    
    summary['expected_monetary_value'] = ggf.conditional_expected_average_profit(
        summary['frequency'],
        summary['monetary_value']
    )
    
    # For customers with 0 frequency, expected value is just their single purchase average
    summary['expected_monetary_value'].fillna(summary['monetary_value'], inplace=True)
    
    # Combine predictions
    summary['probabilistic_clv_90d'] = summary['predicted_purchases_90d'] * summary['expected_monetary_value']
    
    # Log models and return features
    with mlflow.start_run(run_name="Probabilistic_Models", nested=True) as run:
        joblib.dump(bgf, "bgf_model.pkl")
        joblib.dump(ggf, "ggf_model.pkl")
        mlflow.log_artifact("bgf_model.pkl", artifact_path="probabilistic_models")
        mlflow.log_artifact("ggf_model.pkl", artifact_path="probabilistic_models")
        
    return summary.reset_index()[['CustomerID', 'predicted_purchases_90d', 'expected_monetary_value', 'probabilistic_clv_90d']]

def add_probabilistic_features_to_main_set():
    """Integrates probabilistic features into the main feature set."""
    raw_data_path = CONFIG['data']['raw_path']
    processed_feature_path = CONFIG['data']['processed_path']
    
    df_raw = pd.read_csv(raw_data_path, parse_dates=['TransactionDate'])
    df_features = pd.read_csv(processed_feature_path)
    
    # The probabilistic models need the full history up to the prediction start
    train_end_date = pd.to_datetime(CONFIG['time_split']['validation_start_date']) - pd.Timedelta(days=90)
    prob_df_train = df_raw[df_raw['TransactionDate'] <= train_end_date]
    
    prob_features = fit_probabilistic_models(prob_df_train)
    
    # Merge into the main feature set
    df_features_enriched = pd.merge(df_features, prob_features, on='CustomerID', how='left')
    df_features_enriched.fillna(0, inplace=True)
    
    # Overwrite the feature file with the enriched version
    df_features_enriched.to_csv(processed_feature_path, index=False)
    print("Probabilistic features have been added to the main feature set.")

if __name__ == "__main__":
    mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
    mlflow.set_experiment(CONFIG['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name="Main_Pipeline_Features") as parent_run:
        add_probabilistic_features_to_main_set()