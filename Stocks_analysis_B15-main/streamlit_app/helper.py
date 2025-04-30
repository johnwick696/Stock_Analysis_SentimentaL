

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime as dt
import os
from pathlib import Path

import pandas as pd

#import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests

from statsmodels.tsa.ar_model import AutoReg

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


ALPHA_VANTAGE_API_KEY = 'ECP7NGZZHBIK99YH'

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
def fetch_stock_history(stock_ticker, api_key=ALPHA_VANTAGE_API_KEY):
    """
    Fetch daily stock price history using Alpha Vantage.

    Returns:
        pd.DataFrame: DataFrame with columns Open, High, Low, Close.
    """
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='compact')

        # Rename columns to standard format
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        })

        return data[["Open", "High", "Low", "Close"]]
    except Exception as e:
        print("Error fetching stock history:", e)
        return pd.DataFrame()

def fetch_stock_info(stock_ticker, api_key=ALPHA_VANTAGE_API_KEY):
    """
    Fetch fundamental stock info using Alpha Vantage's OVERVIEW API.

    Returns:
        dict: Dictionary containing Basic Information and Market Data.
    """
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock_ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()

        if "Symbol" not in data or not data.get("Name"):
            raise ValueError("Invalid symbol or data not found.")

        def safe_get(key):
            return data.get(key, "N/A")

        stock_data_info = {
            "Basic Information": {
                "Symbol": safe_get("Symbol"),
                "Name": safe_get("Name"),
                "Currency": safe_get("Currency"),
                "Exchange": safe_get("Exchange"),
                "Sector": safe_get("Sector"),
                "Industry": safe_get("Industry"),
            },
            "Market Data": {
                "MarketCapitalization": safe_get("MarketCapitalization"),
                "EBITDA": safe_get("EBITDA"),
                "PERatio": safe_get("PERatio"),
                "PEGRatio": safe_get("PEGRatio"),
                "BookValue": safe_get("BookValue"),
                "DividendPerShare": safe_get("DividendPerShare"),
                "DividendYield": safe_get("DividendYield"),
                "EPS": safe_get("EPS"),
                "RevenueTTM": safe_get("RevenueTTM"),
                "ProfitMargin": safe_get("ProfitMargin"),
                "52WeekHigh": safe_get("52WeekHigh"),
                "52WeekLow": safe_get("52WeekLow"),
            }
        }

        return stock_data_info

    except Exception as e:
        print("Error fetching stock info:", e)
        return {}
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





'''
import datetime as dt
import numpy as np
import tensorflow as tf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


''''''def generate_stock_prediction(stock_ticker):
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
''''''

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

'''

