import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import sys
import os

st.set_page_config(
    page_title="Stock News Analysis",
    page_icon="📰",
    layout="wide"
)

# Load CSS safely
css_path = os.path.join(os.path.dirname(__file__), "../designing.css")
if os.path.exists(css_path):
    with open(css_path) as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ designing.css not found")
css_path = os.path.join(os.path.dirname(__file__), "designing.css")



# Add the parent directory to the path to import helper functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import *

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')




st.markdown("""
<style>
body {
    font-family: 'Comic Sans MS', sans-serif;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Function to fetch news for a company
def fetch_company_news(company_name, ticker, max_news=10):
    news_list = []
    
    # Try to fetch news from Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        yahoo_news = stock.news
        
        for item in yahoo_news[:max_news]:
            news_list.append({
                'title': item['title'],
                'publisher': item.get('publisher', 'Yahoo Finance'),
                'link': item['link'],
                'published': datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S'),
                'summary': item.get('summary', ''),
                'source': 'Yahoo Finance'
            })
    except Exception as e:
        st.warning(f"Could not fetch Yahoo Finance news: {e}")
    
    # If we don't have enough news from Yahoo, try Google News API
    if len(news_list) < max_news:
        try:
            # Google News API endpoint
            url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            for item in items[:max_news - len(news_list)]:
                title = item.title.text
                link = item.link.text
                pub_date = item.pubDate.text
                description = item.description.text if item.description else ''
                
                news_list.append({
                    'title': title,
                    'publisher': 'Google News',
                    'link': link,
                    'published': pub_date,
                    'summary': description,
                    'source': 'Google News'
                })
        except Exception as e:
            st.warning(f"Could not fetch Google News: {e}")
    
    return news_list

# Function to analyze sentiment using NLTK VADER
def analyze_sentiment_vader(text):
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        return sentiment
    except Exception as e:
        st.error(f"Error analyzing sentiment with VADER: {e}")
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}

# Function to predict stock movement based on sentiment
def predict_stock_movement(sentiment_scores):
    # Calculate average compound sentiment score
    avg_compound = sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    # Calculate probability of stock rising
    if avg_compound > 0:
        # Map from [0, 1] to [50%, 100%]
        rise_probability = 50 + (avg_compound * 50)
    else:
        # Map from [-1, 0] to [0%, 50%]
        rise_probability = 50 + (avg_compound * 50)
    
    # Determine prediction and confidence
    if rise_probability > 60:
        prediction = "Bullish (Likely Up)"
        direction = "up"
    elif rise_probability < 40:
        prediction = "Bearish (Likely Down)"
        direction = "down"
    else:
        prediction = "Neutral"
        direction = "neutral"
    
    return prediction, rise_probability, avg_compound, direction

# Function to create historical sentiment chart
def create_historical_sentiment_chart(ticker, period='3mo'):
    try:
        # Fetch historical stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        
        # Create a dataframe for sentiment over time
        dates = []
        sentiment_values = []
        
        # Get news for each week in the period
        current_date = datetime.now()
        for i in range(12):  # Get 12 weeks of data
            end_date = current_date - timedelta(days=i*7)
            start_date = end_date - timedelta(days=7)
            
            # Format dates for the query
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # Try to get news for this period
            try:
                # Use the company name and ticker for better results
                company_name = stock.info.get('shortName', ticker)
                news = fetch_company_news(company_name, ticker, max_news=5)
                
                # Analyze sentiment for each news item
                sentiments = []
                for item in news:
                    sentiment = analyze_sentiment_vader(item['title'] + ' ' + item['summary'])
                    sentiments.append(sentiment)
                
                # Calculate average sentiment for this period
                if sentiments:
                    avg_sentiment = sum(s['compound'] for s in sentiments) / len(sentiments)
                    dates.append(start_date)
                    sentiment_values.append(avg_sentiment)
            except Exception as e:
                st.warning(f"Error getting news for period {start_date_str} to {end_date_str}: {e}")
        
        # Create a dataframe with the sentiment data
        sentiment_df = pd.DataFrame({
            'Date': dates,
            'Sentiment': sentiment_values
        })
        
        # Create a figure with stock price and sentiment
        fig = go.Figure()
        
        # Add stock price
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            name='Stock Price',
            line=dict(color='blue')
        ))
        
        # Add sentiment on a secondary y-axis
        fig.add_trace(go.Scatter(
            x=sentiment_df['Date'],
            y=sentiment_df['Sentiment'],
            name='News Sentiment',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        # Update layout for dual y-axis
        fig.update_layout(
            title=f'{ticker} Stock Price and News Sentiment',
            yaxis=dict(
                title='Stock Price',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title='Sentiment Score',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                anchor='x',
                overlaying='y',
                side='right',
                range=[-1, 1]
            ),
            xaxis=dict(title='Date'),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating historical sentiment chart: {e}")
        return None

# Sidebar for stock selection
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

# News settings
st.sidebar.markdown("### **News Settings**")
max_news = st.sidebar.slider("Maximum News to Fetch", 5, 20, 10)
news_period = st.sidebar.selectbox(
    "Historical Sentiment Period",
    ["1mo", "3mo", "6mo", "1y"],
    index=1
)

# Main content
st.markdown("# **📰 Stock News Analysis**")
st.markdown(f"##### **Track news sentiment and predict stock movements for {stock}**")

# Fetch company news
company_name = stock  # Using stock name as company name
news_list = fetch_company_news(company_name, stock_ticker, max_news)

# Display news
st.markdown("## **Latest News**")
if news_list:
    # Analyze sentiment for each news item
    sentiment_scores = []
    for i, news in enumerate(news_list):
        sentiment = analyze_sentiment_vader(news['title'] + ' ' + news['summary'])
        sentiment_scores.append(sentiment)
        
        # Create an expander for each news item
        with st.expander(f"{i+1}. {news['title']} ({news['publisher']}) - {news['published']}"):
            st.markdown(f"**Source:** {news['source']}")
            st.markdown(f"**Published:** {news['published']}")
            st.markdown(f"**Summary:** {news['summary']}")
            st.markdown(f"**Sentiment:** Positive: {sentiment['pos']:.2f}, Negative: {sentiment['neg']:.2f}, Neutral: {sentiment['neu']:.2f}, Compound: {sentiment['compound']:.2f}")
            st.markdown(f"[Read more]({news['link']})")
    
    # Predict stock movement based on sentiment
    prediction, rise_probability, avg_compound, direction = predict_stock_movement(sentiment_scores)
    
    # Display prediction
    st.markdown("## **Sentiment-Based Prediction**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Sentiment", f"{avg_compound:.2f}", delta=None)
    
    with col2:
        st.metric("Prediction", prediction, delta=None)
    
    with col3:
        st.metric("Confidence", f"{rise_probability:.1f}%", delta=None)
    
    # Create a gauge chart for the prediction
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rise_probability,
        title={'text': "Bullish Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': rise_probability
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create and display historical sentiment chart
    st.markdown("## **Historical Sentiment and Stock Price**")
    hist_fig = create_historical_sentiment_chart(stock_ticker, news_period)
    if hist_fig:
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.warning("Could not create historical sentiment chart.")
    
else:
    st.warning("No news found for the selected stock.")
