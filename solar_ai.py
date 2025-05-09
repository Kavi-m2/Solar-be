# solar_ai.py
# Simulates data, trains model, and saves it

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# --- Data Simulation ---
np.random.seed(42)
num_days = 30
samples_per_day = 24
total_samples = num_days * samples_per_day
start_date = datetime(2025, 1, 1)
timestamps = [start_date + timedelta(hours=i) for i in range(total_samples)]
latitude = 25.0
longitude = 55.0
sun_altitude = [max(0, 90 * np.sin(np.pi * ((ts.hour - 6) / 12))) for ts in timestamps]
sun_azimuth = [(180 + 15 * (ts.hour - 12)) % 360 for ts in timestamps]
cloud_index = np.clip(np.random.normal(0.3, 0.2, total_samples), 0, 1)
dust_level = np.zeros(total_samples)
dust = 0
for i in range(total_samples):
    if np.random.rand() < 0.01:
        dust = 0
    else:
        dust += np.random.rand() * 0.01
    dust_level[i] = min(dust, 1.0)
optimal_tilt = np.clip([alt - 10 * cloud_index[i] - 5 * dust_level[i] for i, alt in enumerate(sun_altitude)], 0, 90)
simulated_tilt = [alt + np.random.normal(0, 5) for alt in optimal_tilt]
energy_output = [max(0, np.cos(np.radians(simulated_tilt[i] - sun_altitude[i])) *
                    (1 - cloud_index[i]) * (1 - 0.5 * dust_level[i]) * 1000)
                 for i in range(total_samples)]

df = pd.DataFrame({
    'sun_altitude': sun_altitude,
    'sun_azimuth': sun_azimuth,
    'cloud_index': cloud_index,
    'dust_level': dust_level,
    'optimal_tilt': optimal_tilt
})

# --- Train and Save Model ---
features = df[['sun_altitude', 'sun_azimuth', 'cloud_index', 'dust_level']]
target = df['optimal_tilt']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)
joblib.dump(model, 'optimal_tilt_model.pkl')
print("✅ Model trained and saved as 'optimal_tilt_model.pkl'")
