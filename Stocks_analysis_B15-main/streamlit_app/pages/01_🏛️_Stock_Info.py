import streamlit as st
from helper import *
import cohere
import os

st.set_page_config(
    page_title="Stock Info",
    page_icon="🏛️",
    layout="wide"
)

# Now it's safe to load CSS or other Streamlit features
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

try:
    stock_data_info = fetch_stock_info(stock_ticker)
except:
    st.error("Error: Unable to fetch the stock data. Please try again later.")
    st.stop()


st.markdown("# **🏛️ Stock Information**")

st.markdown(f"### **About {stock}**")



co = cohere.Client('wmBrgevxRZ3eXb992VdiXgelFgGgKSTQ63gd2wt2') # This is your trial API key
response = co.generate(
  model='command',
  prompt=f'Generate a paragraph only in 100 words about {stock} without asking any questions',
  max_tokens=300,
  temperature=0.9,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')
st.write('{}'.format(response.generations[0].text))




st.markdown("### **Basic Information**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame({"Issuer Name": [stock_data_info["Basic Information"]["longName"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Symbol": [stock_ticker]}),
    hide_index=True,
    width=500,
)

col1.dataframe(
    pd.DataFrame({"Currency": [stock_data_info["Basic Information"]["currency"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(pd.DataFrame({"Exchange": [stock_exchange]}), hide_index=True, width=500)

st.markdown("### **Market Data**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame({"Current Price": [stock_data_info["Market Data"]["currentPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Previous Close": [stock_data_info["Market Data"]["previousClose"]]}),
    hide_index=True,
    width=500,
)
col1, col2, col3 = st.columns(3)

col1.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Market Data"]["open"]]}),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame({"Day Low": [stock_data_info["Market Data"]["dayLow"]]}),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Market Data"]["dayHigh"]]}),
    hide_index=True,
    width=300,
)
col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame(
        {
            "Regular Market Previous Close": [
                stock_data_info["Market Data"]["regularMarketPreviousClose"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Regular Market Open": [stock_data_info["Market Data"]["regularMarketOpen"]]}
    ),
    hide_index=True,
    width=500,
)

col1.dataframe(
    pd.DataFrame(
        {
            "Regular Market Day Low": [
                stock_data_info["Market Data"]["regularMarketDayLow"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Regular Market Day High": [
                stock_data_info["Market Data"]["regularMarketDayHigh"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)

col1, col2, col3 = st.columns(3)

col1.dataframe(
    pd.DataFrame(
        {"Fifty-Two Week Low": [stock_data_info["Market Data"]["fiftyTwoWeekLow"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {"Fifty-Two Week High": [stock_data_info["Market Data"]["fiftyTwoWeekHigh"]]}
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Fifty-Day Average": [stock_data_info["Market Data"]["fiftyDayAverage"]]}
    ),
    hide_index=True,
    width=300,
)
st.markdown("### **Volume and Shares**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame({"Volume": [stock_data_info["Volume and Shares"]["volume"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Regular Market Volume": [
                stock_data_info["Volume and Shares"]["regularMarketVolume"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)

col1, col2, col3 = st.columns(3)

col1.dataframe(
    pd.DataFrame(
        {"Average Volume": [stock_data_info["Volume and Shares"]["averageVolume"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Average Volume (10 Days)": [
                stock_data_info["Volume and Shares"]["averageVolume10days"]
            ]
        }
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {
            "Average Daily Volume (10 Day)": [
                stock_data_info["Volume and Shares"]["averageDailyVolume10Day"]
            ]
        }
    ),
    hide_index=True,
    width=300,
)

col1.dataframe(
    pd.DataFrame(
        {
            "Shares Outstanding": [
                stock_data_info["Volume and Shares"]["sharesOutstanding"]
            ]
        }
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Implied Shares Outstanding": [
                stock_data_info["Volume and Shares"]["impliedSharesOutstanding"]
            ]
        }
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Float Shares": [stock_data_info["Volume and Shares"]["floatShares"]]}
    ),
    hide_index=True,
    width=300,
)

st.markdown("### **Dividends and Yield**")

col1, col2, col3 = st.columns(3)

col1.dataframe(
    pd.DataFrame(
        {"Dividend Rate": [stock_data_info["Dividends and Yield"]["dividendRate"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {"Dividend Yield": [stock_data_info["Dividends and Yield"]["dividendYield"]]}
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Payout Ratio": [stock_data_info["Dividends and Yield"]["payoutRatio"]]}
    ),
    hide_index=True,
    width=300,
)

st.markdown("### **Valuation and Ratios**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame(
        {"Market Cap": [stock_data_info["Valuation and Ratios"]["marketCap"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Enterprise Value": [
                stock_data_info["Valuation and Ratios"]["enterpriseValue"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)

col1.dataframe(
    pd.DataFrame(
        {"Price to Book": [stock_data_info["Valuation and Ratios"]["priceToBook"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Debt to Equity": [stock_data_info["Valuation and Ratios"]["debtToEquity"]]}
    ),
    hide_index=True,
    width=500,
)

col1.dataframe(
    pd.DataFrame(
        {"Gross Margins": [stock_data_info["Valuation and Ratios"]["grossMargins"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Profit Margins": [stock_data_info["Valuation and Ratios"]["profitMargins"]]}
    ),
    hide_index=True,
    width=500,
)

st.markdown("### **Financial Performance**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame(
        {"Total Revenue": [stock_data_info["Financial Performance"]["totalRevenue"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Revenue Per Share": [
                stock_data_info["Financial Performance"]["revenuePerShare"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)

col1, col2, col3 = st.columns(3)

col1.dataframe(
    pd.DataFrame(
        {"Total Cash": [stock_data_info["Financial Performance"]["totalCash"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Total Cash Per Share": [
                stock_data_info["Financial Performance"]["totalCashPerShare"]
            ]
        }
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Total Debt": [stock_data_info["Financial Performance"]["totalDebt"]]}
    ),
    hide_index=True,
    width=300,
)

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame(
        {
            "Earnings Growth": [
                stock_data_info["Financial Performance"]["earningsGrowth"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Revenue Growth": [stock_data_info["Financial Performance"]["revenueGrowth"]]}
    ),
    hide_index=True,
    width=500,
)

col1.dataframe(
    pd.DataFrame(
        {
            "Return on Assets": [
                stock_data_info["Financial Performance"]["returnOnAssets"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Return on Equity": [
                stock_data_info["Financial Performance"]["returnOnEquity"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)

st.markdown("### **Cash Flow**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame({"Free Cash Flow": [stock_data_info["Cash Flow"]["freeCashflow"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Operating Cash Flow": [stock_data_info["Cash Flow"]["operatingCashflow"]]}
    ),
    hide_index=True,
    width=500,
)
st.markdown("### **Analyst Targets**")

col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame(
        {"Target High Price": [stock_data_info["Analyst Targets"]["targetHighPrice"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Target Low Price": [stock_data_info["Analyst Targets"]["targetLowPrice"]]}
    ),
    hide_index=True,
    width=500,
)

col1.dataframe(
    pd.DataFrame(
        {"Target Mean Price": [stock_data_info["Analyst Targets"]["targetMeanPrice"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Target Median Price": [
                stock_data_info["Analyst Targets"]["targetMedianPrice"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
