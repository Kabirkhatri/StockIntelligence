"""
Stock data fetching and processing for Indian markets
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import requests
from config.settings import NSE_SUFFIX, BSE_SUFFIX, POPULAR_STOCKS
from utils.helpers import add_indian_suffix

class StockDataFetcher:
    def __init__(self):
        self.cache_duration = 300  # 5 minutes
    
    def fetch_stock_data(self, symbol, period="1y", exchange="NSE"):
        """
        Fetch stock data from yfinance for Indian markets
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            exchange: 'NSE' or 'BSE'
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV
        """
        try:
            # Add appropriate suffix for Indian stocks
            full_symbol = add_indian_suffix(symbol, exchange)
            
            # Fetch data using yfinance
            ticker = yf.Ticker(full_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for {symbol}. Please check the symbol.")
                return None
            
            # Clean and prepare data
            data = data.round(2)
            data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_stock_info(self, symbol, exchange="NSE"):
        """
        Fetch detailed stock information and fundamentals
        
        Args:
            symbol: Stock symbol
            exchange: 'NSE' or 'BSE'
        
        Returns:
            dict: Stock information
        """
        try:
            full_symbol = add_indian_suffix(symbol, exchange)
            ticker = yf.Ticker(full_symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'current_price': info.get('currentPrice', 0),
                'target_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationMean', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return fundamentals
            
        except Exception as e:
            st.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def fetch_multiple_stocks(self, symbols, period="1y", exchange="NSE"):
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Time period
            exchange: 'NSE' or 'BSE'
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Fetching data for {symbol}...")
            data = self.fetch_stock_data(symbol, period, exchange)
            if data is not None:
                stock_data[symbol] = data
            
            progress_bar.progress((i + 1) / len(symbols))
        
        status_text.text("Data fetching completed!")
        progress_bar.empty()
        status_text.empty()
        
        return stock_data
    
    def get_nifty50_constituents(self):
        """
        Get Nifty 50 constituent stocks
        This is a simplified list - in production, you'd fetch from NSE API
        """
        return POPULAR_STOCKS
    
    def fetch_market_indices(self):
        """
        Fetch major Indian market indices
        """
        indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK',
            'NIFTY IT': '^CNXIT',
            'NIFTY PHARMA': '^CNXPHARMA'
        }
        
        index_data = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                if not data.empty:
                    index_data[name] = data
            except Exception as e:
                st.warning(f"Could not fetch data for {name}: {str(e)}")
        
        return index_data
    
    def get_real_time_price(self, symbol, exchange="NSE"):
        """
        Get real-time or latest available price
        """
        try:
            full_symbol = add_indian_suffix(symbol, exchange)
            ticker = yf.Ticker(full_symbol)
            
            # Get latest data
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                # Fallback to daily data
                data = ticker.history(period="1d")
                if not data.empty:
                    return data['Close'].iloc[-1]
                    
        except Exception as e:
            st.error(f"Error fetching real-time price for {symbol}: {str(e)}")
        
        return None
    
    def search_stocks(self, query):
        """
        Search for stocks based on query
        This is a simplified implementation
        """
        matching_stocks = [stock for stock in POPULAR_STOCKS 
                          if query.upper() in stock.upper()]
        return matching_stocks[:10]  # Return top 10 matches

# Create global instance
stock_fetcher = StockDataFetcher()
