# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allow React frontend to call this API

# Load the trained model
model = joblib.load('optimal_tilt_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # prepare features for prediction
        features = np.array([
            data['sun_altitude'],
            data['sun_azimuth'],
            data['cloud_index'],
            data['dust_level']
        ]).reshape(1, -1)

        # get model output
        tilt = float(model.predict(features)[0])

        # compute estimated energy
        energy_output = max(
            0.0,
            np.cos(np.radians(tilt - data['sun_altitude'])) *
            (1 - data['cloud_index']) *
            (1 - 0.5 * data['dust_level']) * 1000
        )

        return jsonify({
            'optimal_tilt': round(tilt, 2),
            'energy_output': round(energy_output, 2)
        })
    except Exception as e:
        # in case of bad input or other errors
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
