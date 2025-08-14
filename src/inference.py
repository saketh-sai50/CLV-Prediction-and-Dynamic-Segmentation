import mlflow
import pandas as pd
import joblib

# Load models and scaler from a production stage in MLflow
clv_model = mlflow.lightgbm.load_model(f"models:/{CONFIG['models']['clv_model_name']}/Production")
segmentation_model = mlflow.sklearn.load_model(f"models:/{CONFIG['models']['segmentation_model_name']}/Production")
scaler = joblib.load("segmentation_scaler.pkl") # This needs to be saved from training

def predict_clv(df_features):
    """Predicts CLV for a dataframe of features."""
    return clv_model.predict(df_features)

def predict_segment(df_features):
    """Predicts segment for a dataframe of features."""
    # Ensure only features used for clustering are passed and scaled
    cluster_features = ['Recency', 'Frequency', 'MonetaryValue', 'probabilistic_clv_90d']
    X_scaled = scaler.transform(df_features[cluster_features])
    return segmentation_model.predict(X_scaled)