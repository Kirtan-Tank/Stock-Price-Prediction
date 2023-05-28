import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set the seed for reproducibility
np.random.seed(42)

# Function to preprocess the stock data
def preprocess_stock_data(stock_data):
    data = stock_data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to split the data into input and output
def split_data(data):
    X, y = data[:-1], data[1:]
    return X, y

# Function to create the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict stock prices
def predict_stock_prices(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

# Function to calculate metrics
def calculate_metrics(predicted_prices, y_test, scaler):
    mse = np.mean((predicted_prices - scaler.inverse_transform(y_test))**2)
    total_variation = np.sum((scaler.inverse_transform(y_test) - np.mean(scaler.inverse_transform(y_test)))**2)
    explained_variation = np.sum((predicted_prices - scaler.inverse_transform(y_test))**2)
    accuracy = 1 - (explained_variation / total_variation)
    return mse, accuracy

# Streamlit web app
def main():
    st.title('Stock Price Prediction')

    # Input stock symbol
    symbol = st.text_input('Enter the stock symbol (e.g., AAPL)')

    # Load historical stock data using Yahoo Finance API
    stock_data = yf.download(symbol)

    # Preprocess the stock data
    scaled_data, scaler = preprocess_stock_data(stock_data)

    # Split the data into input and output
    X, y = split_data(scaled_data)

    # Reshape the input data
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Create the LSTM model
    model = create_model(input_shape=(X.shape[1], 1))

    # Train the LSTM model
    model.fit(X, y, epochs=5, batch_size=32)

    # Stock price prediction
    st.subheader('Stock Price Prediction')

    # Select start and end dates
    start_date = st.date_input('Select start date')
    end_date = st.date_input('Select end date')

    # Filter stock data based on selected dates
    filtered_data = stock_data[(stock_data.index >= str(start_date)) & (stock_data.index <= str(end_date))]

    # Preprocess the filtered stock data
    filtered_scaled_data, _ = preprocess_stock_data(filtered_data)

    # Split the filtered data into input and output
    filtered_X, filtered_y = split_data(filtered_scaled_data)

    # Reshape the input data
    filtered_X = np.reshape(filtered_X, (filtered_X.shape[0], filtered_X.shape[1], 1))

    # Predict stock prices
    predicted_prices = predict_stock_prices(model, filtered_X, scaler)

