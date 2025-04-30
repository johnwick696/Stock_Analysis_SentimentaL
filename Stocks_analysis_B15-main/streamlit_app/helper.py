

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime as dt
import os
from pathlib import Path

import pandas as pd

#import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.exceptions import AlphaVantageException

from statsmodels.tsa.ar_model import AutoReg

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

api_key = 'ECP7NGZZHBIK99YH'

def fetch_stocks():
    csv_path = Path(__file__).resolve().parent / "../data/equity_issuers.csv"
    df = pd.read_csv(csv_path)
    df = df[["Security Code", "Issuer Name"]]
    stock_dict = dict(zip(df["Security Code"], df["Issuer Name"]))
    return stock_dict


def fetch_periods_intervals():
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
    return periods
def safe_get(data_dict, key):
    return data_dict.get(key, "N/A")

# Fetch stock info from Alpha Vantage
def fetch_stock_info(stock_ticker):
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    try:
        # Fetch the stock data info (e.g., daily data, metadata)
        data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='compact')

        stock_data_info = {
            "Basic Information": {
                "symbol": stock_ticker,
                "longName": safe_get(meta_data, "2. Symbol"),
                "currency": "USD",  # Alpha Vantage doesn't return currency directly, so assuming USD
                "exchange": "N/A",  # No exchange info directly available in Alpha Vantage
            },
            "Market Data": {
                "currentPrice": safe_get(data.iloc[-1], "4. close"),  # Using the last closing price
                "previousClose": safe_get(data.iloc[-2], "4. close"),  # Previous day's close price
                "open": safe_get(data.iloc[-1], "1. open"),
                "dayLow": safe_get(data.iloc[-1], "3. low"),
                "dayHigh": safe_get(data.iloc[-1], "2. high"),
                "fiftyTwoWeekLow": "N/A",  # Alpha Vantage doesn't directly provide 52-week low
                "fiftyTwoWeekHigh": "N/A",  # Alpha Vantage doesn't directly provide 52-week high
                "fiftyDayAverage": "N/A",  # You can compute this if needed using rolling averages
                "twoHundredDayAverage": "N/A",  # You can compute this if needed using rolling averages
            },
            "Volume and Shares": {
                "volume": safe_get(data.iloc[-1], "5. volume"),
                "regularMarketVolume": "N/A",  # Alpha Vantage doesn't provide regular market volume
                "averageVolume": "N/A",  # Alpha Vantage doesn't provide average volume directly
                "sharesOutstanding": "N/A",  # This data is not available from Alpha Vantage
            },
            # The rest of the sections could be adjusted similarly
        }

        return stock_data_info

    except Exception as e:
        print(f"Error fetching stock info: {e}")
        return None

# Fetch stock history (OHLC) from Alpha Vantage
def fetch_stock_history(stock_ticker, period='1d', interval='1min'):
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    try:
        # Fetch historical stock data based on the requested interval
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            data, meta_data = ts.get_intraday(symbol=stock_ticker, interval=interval, outputsize='full')
        else:
            data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='full')

        # Filter columns to match the desired output (Open, High, Low, Close)
        stock_data_history = data[['1. open', '2. high', '3. low', '4. close']]

        # Rename columns to match the original format
        stock_data_history.columns = ['Open', 'High', 'Low', 'Close']

        return stock_data_history

    except Exception as e:
        print(f"Error fetching stock history: {e}")
        return None
'''
def fetch_stock_info(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    time.sleep(5)
    stock_data_info = stock_data.info

    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")

    stock_data_info = {
        "Basic Information": {
            "symbol": safe_get(stock_data_info, "symbol"),
            "longName": safe_get(stock_data_info, "longName"),
            "currency": safe_get(stock_data_info, "currency"),
            "exchange": safe_get(stock_data_info, "exchange"),
        },
        "Market Data": {
            "currentPrice": safe_get(stock_data_info, "currentPrice"),
            "previousClose": safe_get(stock_data_info, "previousClose"),
            "open": safe_get(stock_data_info, "open"),
            "dayLow": safe_get(stock_data_info, "dayLow"),
            "dayHigh": safe_get(stock_data_info, "dayHigh"),
            "regularMarketPreviousClose": safe_get(
                stock_data_info, "regularMarketPreviousClose"
            ),
            "regularMarketOpen": safe_get(stock_data_info, "regularMarketOpen"),
            "regularMarketDayLow": safe_get(stock_data_info, "regularMarketDayLow"),
            "regularMarketDayHigh": safe_get(stock_data_info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow": safe_get(stock_data_info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": safe_get(stock_data_info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage": safe_get(stock_data_info, "fiftyDayAverage"),
            "twoHundredDayAverage": safe_get(stock_data_info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume": safe_get(stock_data_info, "volume"),
            "regularMarketVolume": safe_get(stock_data_info, "regularMarketVolume"),
            "averageVolume": safe_get(stock_data_info, "averageVolume"),
            "averageVolume10days": safe_get(stock_data_info, "averageVolume10days"),
            "averageDailyVolume10Day": safe_get(
                stock_data_info, "averageDailyVolume10Day"
            ),
            "sharesOutstanding": safe_get(stock_data_info, "sharesOutstanding"),
            "impliedSharesOutstanding": safe_get(
                stock_data_info, "impliedSharesOutstanding"
            ),
            "floatShares": safe_get(stock_data_info, "floatShares"),
        },
        "Dividends and Yield": {
            "dividendRate": safe_get(stock_data_info, "dividendRate"),
            "dividendYield": safe_get(stock_data_info, "dividendYield"),
            "payoutRatio": safe_get(stock_data_info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": safe_get(stock_data_info, "marketCap"),
            "enterpriseValue": safe_get(stock_data_info, "enterpriseValue"),
            "priceToBook": safe_get(stock_data_info, "priceToBook"),
            "debtToEquity": safe_get(stock_data_info, "debtToEquity"),
            "grossMargins": safe_get(stock_data_info, "grossMargins"),
            "profitMargins": safe_get(stock_data_info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue": safe_get(stock_data_info, "totalRevenue"),
            "revenuePerShare": safe_get(stock_data_info, "revenuePerShare"),
            "totalCash": safe_get(stock_data_info, "totalCash"),
            "totalCashPerShare": safe_get(stock_data_info, "totalCashPerShare"),
            "totalDebt": safe_get(stock_data_info, "totalDebt"),
            "earningsGrowth": safe_get(stock_data_info, "earningsGrowth"),
            "revenueGrowth": safe_get(stock_data_info, "revenueGrowth"),
            "returnOnAssets": safe_get(stock_data_info, "returnOnAssets"),
            "returnOnEquity": safe_get(stock_data_info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow": safe_get(stock_data_info, "freeCashflow"),
            "operatingCashflow": safe_get(stock_data_info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice": safe_get(stock_data_info, "targetHighPrice"),
            "targetLowPrice": safe_get(stock_data_info, "targetLowPrice"),
            "targetMeanPrice": safe_get(stock_data_info, "targetMeanPrice"),
            "targetMedianPrice": safe_get(stock_data_info, "targetMedianPrice"),
        },
    }

    return stock_data_info


def fetch_stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]
    return stock_data_history

'''
import datetime as dt
import numpy as np
import tensorflow as tf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


'''def generate_stock_prediction(stock_ticker):
    try:
        # Fetch stock data
        stock_data = yf.Ticker(stock_ticker)
        stock_data_hist = stock_data.history(period="2y", interval="1d")
        stock_data_close = stock_data_hist[["Close"]]
        stock_data_close = stock_data_close.asfreq("D", method="ffill").fillna(method="ffill")

        # Train-test split
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]

        # AutoReg Model (AR)
        model = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")
        predictions_ar = model.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)
        forecast_ar = model.predict(start=test_df.index[0], end=test_df.index[-1] + dt.timedelta(days=90), dynamic=True)

        # Scaling the data for LSTM and RNN models
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data_close)

        # Prepare LSTM and RNN training data
        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i : i + time_step])
                y.append(data[i + time_step])
            return np.array(X), np.array(y)

        time_step = 60
        train_size = int(len(scaled_data) * 0.9)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape data for LSTM/RNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM model
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mean_squared_error")
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
        lstm_predictions = lstm_model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)

        # RNN model
        rnn_model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=(time_step, 1)),
            tf.keras.layers.SimpleRNN(100),
            tf.keras.layers.Dense(1),
        ])
        rnn_model.compile(optimizer="adam", loss="mean_squared_error")
        rnn_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
        rnn_predictions = rnn_model.predict(X_test)
        rnn_predictions = scaler.inverse_transform(rnn_predictions)

        return train_df, test_df, forecast_ar, lstm_predictions, rnn_predictions
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None
'''

import datetime as dt
import yfinance as yf
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def generate_stock_prediction(stock_ticker):
    try:
        # Fetch stock data
        stock_data = yf.Ticker(stock_ticker)
        stock_data_hist = stock_data.history(period="2y", interval="1d")
        stock_data_close = stock_data_hist[["Close"]]

        # Fill missing data
        stock_data_close = stock_data_close.asfreq("D", method="ffill").fillna(method="ffill")

        # Train-test split for AutoReg
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]

        # AutoReg Model (AR)
        model_ar = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")
        predictions_ar = model_ar.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)
        forecast_ar = model_ar.predict(start=test_df.index[0], end=test_df.index[-1] + dt.timedelta(days=90), dynamic=True)

        # MinMax scaling for LSTM and RNN
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data_close)

        # Prepare training and testing datasets for LSTM and RNN
        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i : i + time_step])
                y.append(data[i + time_step])
            return np.array(X), np.array(y)

        time_step = 60
        train_size = int(len(scaled_data) * 0.9)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape data for LSTM/RNN models
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM model
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mean_squared_error")
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
        lstm_predictions = lstm_model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)

        # RNN model
        rnn_model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=(time_step, 1)),
            tf.keras.layers.SimpleRNN(100),
            tf.keras.layers.Dense(1),
        ])
        rnn_model.compile(optimizer="adam", loss="mean_squared_error")
        rnn_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
        rnn_predictions = rnn_model.predict(X_test)
        rnn_predictions = scaler.inverse_transform(rnn_predictions)
        rnn_predictions_flat = rnn_predictions.flatten()
        actual_values = test_df["Close"].values
        # Calculate RMSE
        def calculate_rmse(actual, predicted):
            return np.sqrt(mean_squared_error(actual, predicted))

        rnn_rmse = calculate_rmse(actual_values, predictions_ar)

        # Calculate MAPE (Mean Absolute Percentage Error)
        def calculate_mape(actual, predicted):
                return np.mean(np.abs((actual - predicted) / actual)) * 100

        rnn_mape = calculate_mape(actual_values, predictions_ar)

        
        return train_df, test_df, forecast_ar, lstm_predictions, rnn_predictions, predictions_ar ,rnn_mape

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None


