.PHONY: all setup data features train serve-api serve-dashboard validate-data promote-clv promote-segment

# Default command
all:
	@echo "Available commands: setup, data, features, train, serve-api, serve-dashboard"

# Setup virtual environment and install dependencies
setup:
	python3 -m venv venv
	@. venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	@echo "Setup complete. Activate the environment with: source venv/bin/activate"

# Step 1: Incremental data load and validation
data:
	python -c "from src.incremental_loader import load_new_data; load_new_data()"

# Step 1.5: Run validation suite manually
validate-data:
	great_expectations suite edit expectations/transaction_suite.json

# Step 2: Build feature sets
features:
	python -c "from src.feature_engineering import build_feature_set; build_feature_set()"
	python -c "from src.probabilistic import add_probabilistic_features_to_main_set; add_probabilistic_features_to_main_set()"

# Step 3: Train all models and register them
train:
	python -c "from src.utils_io import load_config; import mlflow; CONFIG = load_config(); mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri']); mlflow.set_experiment(CONFIG['mlflow']['experiment_name']); \
	with mlflow.start_run(run_name='Main_Training_Pipeline') as parent_run: \
		from src.train_regression import train_clv_model; train_clv_model(); \
		from src.train_segmentation import train_segmentation_models; train_segmentation_models()"

# Step 4.1: Serve Flask API
serve-api:
	python api/app.py

# Step 4.2: Serve Streamlit Dashboard
serve-dashboard:
	streamlit run dashboard/app.py

# Bonus: MLflow Model Promotion Commands
promote-clv:
	# Promotes version 1 of the CLV model to Production
	mlflow models transition-stage \
		name="production-clv-predictor" \
		version=1 \
		stage="Production"

promote-segment:
	# Promotes version 1 of the segmentation model to Production
	mlflow models transition-stage \
		name="production-customer-segmenter" \
		version=1 \
		stage="Production"