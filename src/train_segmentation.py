import pandas as pd
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import mlflow
import joblib
from src.utils_io import load_config
from src.labeler import assign_segment_labels

CONFIG = load_config()
mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
mlflow.set_experiment(CONFIG['mlflow']['experiment_name'])

def train_segmentation_models():
    """Trains, compares, and registers the best segmentation model."""
    print("Training and comparing segmentation models...")
    df = pd.read_csv(CONFIG['data']['processed_path'])
    
    # Use a subset of features for interpretable clustering
    feature_cols = ['Recency', 'Frequency', 'MonetaryValue', 'probabilistic_clv_90d']
    X = df[feature_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_model_info = {'name': None, 'score': -1, 'run_id': None, 'model': None, 'k': 0}
    
    with mlflow.start_run(run_name="Segmentation_Comparison", nested=True):
        joblib.dump(scaler, "segmentation_scaler.pkl")
        mlflow.log_artifact("segmentation_scaler.pkl")

        for k in CONFIG['segmentation_params']['n_clusters_range']:
            # K-Means
            with mlflow.start_run(run_name=f"KMeans_k={k}", nested=True) as run:
                kmeans = KMeans(n_clusters=k, random_state=CONFIG['seeds']['model_training_seed'], n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, clusters)
                mlflow.log_params({"model": "KMeans", "k": k})
                mlflow.log_metric("silhouette", score)
                if score > best_model_info['score']:
                    best_model_info.update({'name': 'KMeans', 'score': score, 'run_id': run.info.run_id, 'model': kmeans, 'k': k})
            
            # Gaussian Mixture Model
            with mlflow.start_run(run_name=f"GMM_k={k}", nested=True) as run:
                gmm = GaussianMixture(n_components=k, random_state=CONFIG['seeds']['model_training_seed'])
                clusters = gmm.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, clusters)
                mlflow.log_params({"model": "GMM", "k": k})
                mlflow.log_metric("silhouette", score)
                if score > best_model_info['score']:
                    best_model_info.update({'name': 'GMM', 'score': score, 'run_id': run.info.run_id, 'model': gmm, 'k': k})
    
    print(f"Best model is {best_model_info['name']} with k={best_model_info['k']} (Silhouette: {best_model_info['score']:.3f})")

    # Log and Register the BEST model
    best_run_id = best_model_info['run_id']
    best_model = best_model_info['model']
    
    with mlflow.start_run(run_id=best_run_id):
        mlflow.sklearn.log_model(best_model, "segmentation_model")
        model_uri = f"runs:/{best_run_id}/segmentation_model"
        mlflow.register_model(model_uri, CONFIG['models']['segmentation_model_name'])
        print(f"Best model registered as '{CONFIG['models']['segmentation_model_name']}'")

if __name__ == "__main__":
    with mlflow.start_run(run_name="Main_Training_Pipeline") as parent_run:
        train_segmentation_models()