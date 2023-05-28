import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("modified/saved_model.h5")

# Set the page title
st.title('Stock Price Prediction')

# Get user input for stock symbol
stock_symbol = st.text_input('Enter stock symbol')

# Get user input for start and end dates
start_date = st.date_input('Enter start date')
end_date = st.date_input('Enter end date')

# Convert dates to the required format
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Fetch stock price data using yfinance
stock_data = yf.download(stock_symbol, start=start_date_str, end=end_date_str)

# Check if data is available for the given stock symbol and date range
if stock_data.empty:
    st.write('No data available for the given stock symbol and date range.')
else:
    # Perform data preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[['Close']])

    # Prepare the input features
    # Here, you can perform additional feature engineering or use the desired features for prediction
    X = scaled_data[:-1]  # Exclude the last row
    X = X.reshape(1, -1)  # Reshape to match the model's input shape

    # Make predictions using the pre-trained model
    predictions = model.predict(X)

    # Inverse transform the predictions to get the actual prices
    actual_predictions = scaler.inverse_transform(predictions)

    # Display the predicted price
    predicted_price = actual_predictions[0][0]
    st.write('Predicted Price:', predicted_price)

    # Plot the actual prices over time
    st.line_chart(stock_data['Close'])
