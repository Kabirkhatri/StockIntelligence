"""
Configuration settings for the stock analysis platform
"""

# Indian Stock Market Settings
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

# Popular Indian stocks
POPULAR_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL",
    "KOTAKBANK", "LT", "ASIANPAINT", "MARUTI", "BAJFINANCE", "HCLTECH", "WIPRO",
    "ULTRACEMCO", "SUNPHARMA", "TITAN", "TECHM", "POWERGRID", "NESTLEIND", "ONGC",
    "TATAMOTORS", "NTPC", "AXISBANK", "BAJAJFINSV", "ICICIBANK", "COALINDIA", "ADANIPORTS"
]

# Technical Analysis Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Reinforcement Learning Parameters
RL_LOOKBACK_WINDOW = 30
RL_ACTIONS = ['BUY', 'SELL', 'HOLD']
RL_EPISODES = 1000
RL_LEARNING_RATE = 0.001
RL_GAMMA = 0.95
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.01
RL_EPSILON_DECAY = 0.995

# Portfolio Settings
INITIAL_CAPITAL = 100000  # 1 Lakh INR
MAX_POSITION_SIZE = 0.1  # 10% of portfolio
TRANSACTION_COST = 0.001  # 0.1%

# News API Settings (placeholder for news sources)
NEWS_SOURCES = [
    "https://economictimes.indiatimes.com/",
    "https://www.moneycontrol.com/",
    "https://www.business-standard.com/"
]
