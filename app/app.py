from flask import Flask, request, jsonify
import pickle
import pandas as pd
import sys
import os


# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import preprocess_input  # import your function

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return 'Telco Churn Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input
        processed_input = preprocess_input(input_df)  # use your function here
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        result = 'Churn' if prediction == 1 else 'No Churn'
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
