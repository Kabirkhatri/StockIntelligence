"""
Helper utilities for the stock analysis platform
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def format_currency(amount):
    """Format currency in Indian format"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f}L"
    else:
        return f"₹{amount:,.2f}"

def calculate_returns(prices):
    """Calculate daily returns from price series"""
    return prices.pct_change().dropna()

def calculate_volatility(returns, window=30):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_sharpe_ratio(returns, risk_free_rate=0.06):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate/252
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def add_indian_suffix(symbol, exchange="NSE"):
    """Add appropriate suffix for Indian stocks"""
    if exchange == "NSE":
        return f"{symbol}.NS"
    elif exchange == "BSE":
        return f"{symbol}.BO"
    return symbol

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_data(func, *args, **kwargs):
    """Generic caching wrapper"""
    return func(*args, **kwargs)

def validate_stock_symbol(symbol):
    """Validate if stock symbol is valid"""
    if not symbol or len(symbol) < 2:
        return False
    return symbol.replace('.NS', '').replace('.BO', '').isalpha()

def create_date_range(start_date, end_date):
    """Create date range for analysis"""
    return pd.date_range(start=start_date, end=end_date, freq='D')

def normalize_data(data, method='minmax'):
    """Normalize data for ML models"""
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    return data

def calculate_portfolio_metrics(returns):
    """Calculate comprehensive portfolio metrics"""
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown((1 + returns).cumprod())
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }
