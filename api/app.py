from flask import Flask, request, jsonify
import pandas as pd
from src.inference import predict_clv, predict_segment # Simplified, needs path adjustment
from src.utils_io import load_config
from src.labeler import assign_segment_labels
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)
CONFIG = load_config()

# A placeholder for a feature store lookup
ALL_FEATURES_DF = pd.read_csv(CONFIG['data']['processed_path'])
ALL_FEATURES_DF.set_index('CustomerID', inplace=True)

@app.route('/predict-clv', methods=['POST'])
def handle_clv():
    data = request.get_json()
    customer_ids = data.get('customer_ids', [])
    
    if not customer_ids:
        return jsonify({"error": "customer_ids not provided"}), 400
        
    try:
        customer_features = ALL_FEATURES_DF.loc[customer_ids]
        clv_preds = predict_clv(customer_features)
        response = dict(zip(customer_ids, clv_preds))
        return jsonify(response)
    except KeyError:
        return jsonify({"error": "One or more customer_ids not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-segment', methods=['POST'])
def handle_segment():
    data = request.get_json()
    customer_ids = data.get('customer_ids', [])

    if not customer_ids:
        return jsonify({"error": "customer_ids not provided"}), 400

    try:
        customer_features = ALL_FEATURES_DF.loc[customer_ids].copy()
        segments = predict_segment(customer_features)
        customer_features['segment'] = segments
        
        # Assign business labels
        labeled_df, label_map = assign_segment_labels(customer_features.reset_index())
        
        response = labeled_df[['CustomerID', 'segment', 'segment_label']].to_dict(orient='records')
        return jsonify({"predictions": response, "label_mapping": label_map})
    except KeyError:
        return jsonify({"error": "One or more customer_ids not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)