import plotly.graph_objects as go
import streamlit as st

from helper import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
)


with open("/Users/labuser/Documents/Stocks_analysis_B15-main/streamlit_app/designing.css") as source_des:
    st.markdown(f'<style>{source_des.read()}</style>', unsafe_allow_html=True)


st.markdown("""
<style>
body {
    font-family: 'Comic Sans MS', sans-serif;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## **User Input Features**")
stock_dict = fetch_stocks()
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))
st.sidebar.markdown("### **Select stock exchange**")
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)
stock_ticker = f"{stock_dict[stock]}.{'BO' if stock_exchange == 'BSE' else 'NS'}"
st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(
    label="Stock ticker code", placeholder=stock_ticker, disabled=True
)
periods = fetch_periods_intervals()
st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))
st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])




st.markdown("# **📈 Stock Prediction**")
st.markdown(f"##### **Optimize investment strategies with data-driven forecasting specifically tailored for {stock}**")


stock_data = fetch_stock_history(stock_ticker, period, interval)
st.markdown("## **Historical Data**")
fig = go.Figure(
    data=[
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
        )
    ]
)
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)


train_df, test_df, forecast_ar, lstm_predictions, rnn_predictions, predictions_ar,rnn_mape = generate_stock_prediction(stock_ticker)

if train_df is not None and (forecast_ar >= 0).all() and (predictions_ar >= 0).all() and (lstm_predictions >= 0).all() and (rnn_predictions >= 0).all():
    st.markdown("## **Stock Prediction**")
    
    # Create the figure with all the models' predictions
    fig = go.Figure(
        data=[
            # Plot training data
            go.Scatter(
                x=train_df.index,
                y=train_df["Close"],
                name="Train",
                mode="lines",
                line=dict(color="blue"),
            ),
            # Plot actual test data
            go.Scatter(
                x=test_df.index,
                y=test_df["Close"],
                name="Test",
                mode="lines",
                line=dict(color="orange"),
            ),
            # Plot forecast from AutoReg
            go.Scatter(
                x=forecast_ar.index,
                y=forecast_ar,
                name="Forecast",
                mode="lines",
                line=dict(color="red"),
            ),
            # Plot test predictions from AutoReg
            go.Scatter(
                x=test_df.index,
                y=predictions_ar,
                name="Test Predictions",
                mode="lines",
                line=dict(color="green"),
            ),
            # Plot LSTM predictions
            go.Scatter(
                x=test_df.index,
                y=lstm_predictions.flatten(),
                name="LSTM Predictions",
                mode="lines",
                line=dict(color="purple"),
            ),
            # Plot RNN predictions
            go.Scatter(
                x=test_df.index,
                y=rnn_predictions.flatten(),
                name="RNN Predictions",
                mode="lines",
                line=dict(color="cyan"),
            ),
        ]
    )
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**MAPE (Mean Absolute Percentage Error)**: {rnn_mape}%")
    st.markdown(f"**Accuracy**: {100-rnn_mape}%")
else:
    st.markdown("## **Stock Prediction**")
    st.markdown("### **No data available for the selected stock**")
