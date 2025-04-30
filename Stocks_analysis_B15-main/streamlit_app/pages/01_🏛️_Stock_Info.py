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

# Utility: Display info in columns
def show_dataframe(label, value, col):
    df = pd.DataFrame({label: [value]})
    col.dataframe(df, hide_index=True, width=500 if col == col1 or col == col2 else 300)

# Display Sections
def render_section(title, fields, layout):
    st.markdown(f"### **{title}**")
    for group in layout:
        cols = st.columns(len(group))
        for (field_key, label), col in zip(group, cols):
            try:
                show_dataframe(label, stock_data_info[title][field_key], col)
            except KeyError:
                show_dataframe(label, "N/A", col)

# Display All Sections
render_section("Basic Information", 
    ["longName", "currency"],
    [[("longName", "Issuer Name"), ("ticker", "Symbol")],
     [("currency", "Currency"), ("exchange", "Exchange")]])

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
