#!python3 -m pip install python_deriv_api
#!pip install -U git+https://github.com/mdn522/binaryapi.git
import os
import time
from rich.console import Console
from binaryapi.stable_api import Binary
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
import asyncio
from deriv_api import DerivAPI
from deriv_api import APIError
from datetime import datetime
from binaryapi.constants import CONTRACT_TYPE, DURATION

app_id = 33254
api_token = os.getenv('DERIV_TOKEN', 'GatUOOB4qST3Dum')

if len(api_token) == 0:
    sys.exit("DERIV_TOKEN environment variable is not set")
# Binary Token
token = os.environ.get('BINARY_TOKEN', 'GatUOOB4qST3Dum')

console = Console(log_path=False)
# Derivitives (1HZ10V, 1HZ25V, 1HZ50V, 1HZ75V, 1HZ100V, R_100)
# Forex (frxEURUSD, frxAUDCAD, frxAUDCHF, frxAUDUSD, frxEURGBP)
symbol = "frxEURUSD"
initial_price = 1
duration = 180
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def message_handler(message):
    msg_type = message.get('msg_type')

    if msg_type in ['candles', 'ohlc']:
        # Print candles data from message
        candles_data = message['candles']
        price_data = candles_data

        data = pd.DataFrame(price_data)

        # Load your stock price data
        # Assuming you have a CSV file with an 'Epoch', 'Date', and 'Close' column

        data = pd.DataFrame(price_data)
        data['epoch'] = pd.to_datetime(data['epoch'], unit='s')
        data.set_index('epoch', inplace=True)
        data = data[['close']]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create training dataset
        training_data_len = int(np.ceil(len(scaled_data) * .95))
        train_data = scaled_data[0:int(training_data_len), :]

        # Split the data into x_train and y_train datasets
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data for LSTM model
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # Prepare the last 60 days closing prices for predicting the next candlestick
        last_60_days = scaled_data[-60:].reshape(1, -1, 1)

        # Make predictions for the next candlestick
        predicted_price = model.predict(last_60_days)
        predicted_price = scaler.inverse_transform(predicted_price)

        print(f"Predicted Price for the Next Candlestick: {predicted_price[0, 0]:.2f}")

        if predicted_price > price_data[-1]['close']:
          print("Place Bulish Trade")
          
        else:
          print("Place Bearish Trade")
          

if __name__ == '__main__':
    binary = Binary(token=token, message_callback=message_handler)

    #symbol = 'frxEURUSD'
    #symbol = symbol
    style = 'candles'  # 'ticks' or 'candles'
    end = 'latest'  # 'latest' or unix epoch
    count = 1500  # default: 5000 if not provided
    granularity = 900

    # Subscribe to ticks stream
    binary.api.ticks_history(
        ticks_history=symbol,
        style=style,
        count=count,
        end=end,
        granularity=granularity,
        subscribe=False,
    )

    # Wait for 60 seconds then exit the script
    time.sleep(60)
