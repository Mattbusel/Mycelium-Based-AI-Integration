import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import csv
from datetime import datetime


DATA_PATH = 'path_to_dataset.csv'
MODEL_PATH = 'mycelium_growth_model.pkl'


data = pd.read_csv(DATA_PATH)
print(data.head())


features = ['time', 'temperature', 'humidity', 'ph', 'light_intensity', 'co2_level', 'substrate_type']
target = 'growth'

X = data[features].values
y = data[target].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(),
}

trained_models = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    trained_models[name] = mdl
    print(f"{name} trained.")


cv_scores = {}
for name, mdl in models.items():
    scores = cross_val_score(mdl, X, y, cv=5, scoring='r2')
    cv_scores[name] = scores.mean()
    print(f"{name} CV R²: {scores.mean():.3f}")

best_model_name = max(trained_models, key=lambda name: r2_score(y_test, trained_models[name].predict(X_test)))
best_model = trained_models[best_model_name]
joblib.dump(best_model, MODEL_PATH)
print(f"Saved best model: {best_model_name}")


y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


plt.scatter(X[:, 0], y, color='blue', label='Observed Growth')
plt.scatter(X_test[:, 0], y_pred, color='green', label='Predicted Growth')
plt.xlabel('Time (days)')
plt.ylabel('Growth (cm)')
plt.title('Mycelium Growth Prediction')
plt.legend()
plt.show()


def influence_growth(growth_values, factor):
    return growth_values * factor


def predict_and_influence(new_data, factor=1.0):
    loaded_model = joblib.load(MODEL_PATH)
    pred = loaded_model.predict(new_data)
    influenced = influence_growth(pred, factor)
    return influenced


LOG_FILE = 'model_performance_log.csv'
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


log_entry = {
    'timestamp': timestamp,
    'best_model': best_model_name,
    'test_mse': mse,
    'test_r2': r2,
}

for name, score in cv_scores.items():
    log_entry[f'cv_r2_{name}'] = score


file_exists = os.path.isfile(LOG_FILE)
with open(LOG_FILE, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=log_entry.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(log_entry)

print(f"Logged metrics to {LOG_FILE}")
