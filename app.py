import os
import joblib

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)
API_KEY = os.getenv('API_KEY')

model_path = Path('output/model.joblib')
model = joblib.load(model_path)

@app.route('/', methods=['POST', 'GET'])
def predict_route():
    api_key = request.headers.get('Authorization')

    if api_key != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = pd.DataFrame(request.get_json())
        prediction = predict(data)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def predict(data):
    output=model.predict(data)
    return output[0].item()

if __name__ == "__main__":
    app.run()

    

