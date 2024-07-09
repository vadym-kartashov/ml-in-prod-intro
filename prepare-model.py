import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/train.csv')

X = data['excerpt']
y = data['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000)
X_train_transformed = vectorizer.fit_transform(X_train)
X_val_transformed = vectorizer.transform(X_val)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_val_transformed)
mse = mean_squared_error(y_val, y_pred)
print(f'Mean squared error: {mse}')

joblib.dump(model, 'readability-prediction-model.joblib')
joblib.dump(vectorizer, 'readability-prediction-vectorizer.joblib')