# CLV-Prediction-and-Dynamic-Segmentation

This project implements an end-to-end system for predicting Customer Lifetime Value (CLV) and performing dynamic customer segmentation. It uses a hybrid approach of probabilistic models and machine learning, built with production-minded MLOps practices.

## Project Architecture
- **Data Pipeline**: Features incremental loading and data validation with Great Expectations.
- **Feature Engineering**: Creates RFM and advanced behavioral features.
- **Probabilistic Modeling**: Uses the `lifetimes` library (BG/NBD + Gamma-Gamma) to generate predictive features.
- **Machine Learning**: Trains a LightGBM model for CLV prediction and compares KMeans vs. GMM for segmentation.
- **MLOps**: Leverages MLflow for experiment tracking, model versioning, and registry.
- **Serving**: Deploys models via a Flask API and visualizes insights with a Streamlit dashboard.

## Setup

1.  **Create Virtual Environment:**
    ```bash
    make setup
    ```
2.  **Activate Environment:**
    ```bash
    source venv/bin/activate
    ```

## How to Run the Pipeline

The project is designed to be run sequentially using the `Makefile`.

1.  **Load & Validate Data:**
    This generates synthetic data (first run only), loads new transactions, and validates them.
    ```bash
    make data
    ```

2.  **Build All Features:**
    This creates the main feature set and enriches it with outputs from the probabilistic models.
    ```bash
    make features
    ```

3.  **Train Models:**
    This runs the entire training pipeline, logging experiments and registering the best models to the MLflow registry in the "Staging" stage.
    ```bash
    make train
    ```
    *You can view the results by starting the MLflow UI:*
    ```bash
    mlflow ui --backend-store-uri file:./mlflow_runs
    ```

4.  **Promote Models to Production:**
    After reviewing the models in the MLflow UI, promote them to the "Production" stage so the API and dashboard can use them.
    ```bash
    make promote-clv
    make promote-segment
    ```

5.  **Serve API & Dashboard:**
    Run these commands in separate terminals.

    * **Start the Flask API:**
        ```bash
        make serve-api
        ```
    * **Start the Streamlit Dashboard:**
        ```bash
        make serve-dashboard
        ```

## Example API Calls

Use `curl` to test the running Flask API.

**Request Body (save as `customers.json`):**
```json
{
    "customer_ids": ["C1001", "C1002"]
}
