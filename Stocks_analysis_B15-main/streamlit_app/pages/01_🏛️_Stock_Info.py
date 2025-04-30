import streamlit as st
import pandas as pd
from helper import *
import cohere
import os


# Load environment variables

COHERE_API_KEY = 'wmBrgevxRZ3eXb992VdiXgelFgGgKSTQ63gd2wt2'

st.set_page_config(page_title="Stock Info", page_icon="🏛️", layout="wide")

# Custom CSS
css_path = os.path.join(os.path.dirname(__file__), "../designing.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
body {
    font-family: 'Comic Sans MS', sans-serif;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## **User Input Features**")

# Load stocks
stock_dict = fetch_stocks()

# Sidebar options
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)
stock_ticker = f"{stock_dict[stock]}.{'BO' if stock_exchange == 'BSE' else 'NS'}"
st.sidebar.text_input("Stock ticker code", value=stock_ticker, disabled=True)

# Fetch stock data
try:
    stock_data_info = fetch_stock_info(stock_ticker)
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

st.markdown("# **🏛️ Stock Information**")
st.markdown(f"### **About {stock}**")

# Cohere summary
co = cohere.Client(COHERE_API_KEY)
response = co.generate(
    model='command',
    prompt=f'Generate a paragraph only in 100 words about {stock} without asking any questions',
    max_tokens=300,
    temperature=0.9
)
st.write(response.generations[0].text.strip())
def show_dataframe(label, data, col, width=300): 
    df = pd.DataFrame({label: [data]})
    col.dataframe(df, hide_index=True, width=width)




# Display Sections
def render_section(label, stock_data_info):
    if "Basic Information" in stock_data_info:
        long_name = stock_data_info["Basic Information"].get("longName", "N/A")
    else:
        st.error("Basic Information not found in the stock data.")
        long_name = "N/A"

    # Now display the data in the relevant section (assuming show_dataframe is your function for displaying data)
    show_dataframe(label, long_name, col1, width=500)

# Calling render_section with stock_data_info
render_section("Basic Information", stock_data_info)

    # Continue for the other columns or sections


    # Continue for the other columns or sections



render_section("Market Data",
    [],
    [[("currentPrice", "Current Price"), ("previousClose", "Previous Close")],
     [("open", "Open"), ("dayLow", "Day Low"), ("dayHigh", "Day High")],
     [("regularMarketPreviousClose", "Regular Market Previous Close"), ("regularMarketOpen", "Regular Market Open")],
     [("regularMarketDayLow", "Regular Market Day Low"), ("regularMarketDayHigh", "Regular Market Day High")],
     [("fiftyTwoWeekLow", "52 Week Low"), ("fiftyTwoWeekHigh", "52 Week High"), ("fiftyDayAverage", "50 Day Average")]])

render_section("Volume and Shares",
    [],
    [[("volume", "Volume"), ("regularMarketVolume", "Regular Market Volume")],
     [("averageVolume", "Average Volume"), ("averageVolume10days", "Avg Vol (10D)"), ("averageDailyVolume10Day", "Avg Daily Vol (10D)")],
     [("sharesOutstanding", "Shares Outstanding"), ("impliedSharesOutstanding", "Implied Shares Outstanding"), ("floatShares", "Float Shares")]])

render_section("Dividends and Yield",
    [],
    [[("dividendRate", "Dividend Rate"), ("dividendYield", "Dividend Yield"), ("payoutRatio", "Payout Ratio")]])

render_section("Valuation and Ratios",
    [],
    [[("marketCap", "Market Cap"), ("enterpriseValue", "Enterprise Value")],
     [("priceToBook", "Price to Book"), ("debtToEquity", "Debt to Equity")],
     [("grossMargins", "Gross Margins"), ("profitMargins", "Profit Margins")]])

render_section("Financial Performance",
    [],
    [[("totalRevenue", "Total Revenue"), ("revenuePerShare", "Revenue Per Share")],
     [("totalCash", "Total Cash"), ("totalCashPerShare", "Cash Per Share"), ("totalDebt", "Total Debt")],
     [("earningsGrowth", "Earnings Growth"), ("revenueGrowth", "Revenue Growth")],
     [("returnOnAssets", "Return on Assets"), ("returnOnEquity", "Return on Equity")]])

render_section("Cash Flow",
    [],
    [[("freeCashflow", "Free Cash Flow"), ("operatingCashflow", "Operating Cash Flow")]])

render_section("Analyst Targets",
    [],
    [[("targetHighPrice", "Target High Price"), ("targetLowPrice", "Target Low Price")],
     [("targetMeanPrice", "Target Mean Price"), ("targetMedianPrice", "Target Median Price")]])
