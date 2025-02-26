
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Load Data
def load_data(filename):
    df = pd.read_csv(filename)
    df['datum'] = pd.to_datetime(df['datum'])
    df.sort_values(by='datum', inplace=True)
    df.set_index('datum', inplace=True)
    return df

# Feature Engineering
def feature_engineering(df):
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    return df

# Weighted Moving Average
def weighted_moving_average(series, window_size=3):
    weights = np.arange(1, window_size + 1)
    return series.rolling(window=window_size).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

# Exponential Smoothing
def exponential_smoothing(series, alpha=0.3):
    model = SimpleExpSmoothing(series)
    model_fit = model.fit(smoothing_level=alpha)
    return model_fit.fittedvalues

# Train ML Models
def train_ml_models(X_train, X_test, y_train, y_test):
    models = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    return models

# Hyperparameter Tuning for Random Forest
def tune_random_forest(X_train, y_train):
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# LSTM Model
def train_lstm(X_train, y_train, X_test):
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
    return model.predict(X_test)

# Evaluate Model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return f"{model_name} - MAE: {mae}, RMSE: {rmse}"

# Classification Model for Demand Level
def train_classification_model(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted')

# Main Execution
if __name__ == "__main__":
    st.title("Demand Forecasting Dashboard")
    df = load_data("Sales_project.csv")
    df = feature_engineering(df)
    
    X = df.drop(columns=['Quantity'])
    y = df['Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    df['WMA_Forecast'] = weighted_moving_average(y_train)
    df['ExpSmoothing_Forecast'] = exponential_smoothing(y_train)
    
    models = train_ml_models(X_train, X_test, y_train, y_test)
    best_rf = tune_random_forest(X_train, y_train)
    
    lstm_predictions = train_lstm(X_train, y_train, X_test)
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append(evaluate_model(y_test, y_pred, name))
    
    results.append(evaluate_model(y_test, best_rf.predict(X_test), "Tuned Random Forest"))
    results.append(evaluate_model(y_test, lstm_predictions, "LSTM"))
    
    for res in results:
        st.write(res)
