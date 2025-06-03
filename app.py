"""
Main Streamlit application for Indian Stock Market RL Analysis Platform
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Import custom modules
from data.stock_data import stock_fetcher
from analysis.technical_analysis import technical_analyzer
from analysis.fundamental_analysis import fundamental_analyzer
from analysis.sentiment_analysis import sentiment_analyzer
from ml.rl_model import rl_trading_system
from config.settings import POPULAR_STOCKS
from utils.helpers import format_currency, calculate_portfolio_metrics

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market RL Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #00C851;
    }
    .negative {
        color: #ff4444;
    }
    .neutral {
        color: #ffbb33;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Title and description
    st.title("🚀 Indian Stock Market RL Analysis Platform")
    st.markdown("Advanced reinforcement learning-powered stock analysis for NSE/BSE markets")
    
    # Sidebar
    create_sidebar()
    
    # Main content area
    if 'selected_stock' in st.session_state and st.session_state.selected_stock:
        display_stock_analysis()
    else:
        display_home_page()

def create_sidebar():
    """Create sidebar with navigation and stock selection"""
    
    st.sidebar.title("🎯 Navigation")
    
    # Stock selection
    st.sidebar.subheader("📊 Select Stock")
    
    # Stock search
    search_query = st.sidebar.text_input("Search Stock Symbol", placeholder="e.g., RELIANCE, TCS")
    
    if search_query:
        matching_stocks = stock_fetcher.search_stocks(search_query)
        if matching_stocks:
            selected_stock = st.sidebar.selectbox("Select from matches:", matching_stocks)
        else:
            st.sidebar.warning("No matching stocks found")
            selected_stock = None
    else:
        # Popular stocks dropdown
        selected_stock = st.sidebar.selectbox(
            "Or choose from popular stocks:",
            options=[""] + POPULAR_STOCKS,
            index=0
        )
    
    # Exchange selection
    exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE"], index=0)
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3  # Default to 1 year
    )
    
    # Store selections in session state
    if selected_stock:
        st.session_state.selected_stock = selected_stock
        st.session_state.exchange = exchange
        st.session_state.period = period
        
        # Display current selection
        st.sidebar.success(f"Selected: {selected_stock} ({exchange})")
    
    # Analysis options
    st.sidebar.subheader("🔍 Analysis Options")
    
    st.session_state.show_technical = st.sidebar.checkbox("Technical Analysis", value=True)
    st.session_state.show_fundamental = st.sidebar.checkbox("Fundamental Analysis", value=True)
    st.session_state.show_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    st.session_state.show_rl = st.sidebar.checkbox("RL Predictions", value=True)
    
    # Market indices
    st.sidebar.subheader("📈 Market Indices")
    if st.sidebar.button("Show Market Overview"):
        st.session_state.show_market_overview = True
    
    # Model training
    st.sidebar.subheader("🤖 RL Model")
    if st.sidebar.button("Train RL Model"):
        if 'selected_stock' in st.session_state:
            train_rl_model()
        else:
            st.sidebar.error("Please select a stock first")

def display_home_page():
    """Display home page when no stock is selected"""
    
    st.markdown("## Welcome to the Indian Stock Market RL Analysis Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
        - **Fundamental Analysis**: P/E, ROE, Debt-to-Equity ratios
        - **Sentiment Analysis**: News and social media sentiment
        - **RL Trading**: Deep Q-Network for trading decisions
        - **Portfolio Tracking**: Performance metrics and backtesting
        - **Real-time Data**: NSE/BSE market data
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Getting Started
        1. Select a stock from the sidebar
        2. Choose your preferred exchange (NSE/BSE)
        3. Set the analysis time period
        4. Enable desired analysis modules
        5. View comprehensive analysis results
        """)
    
    # Market overview
    if st.session_state.get('show_market_overview', False):
        display_market_overview()
    
    # Popular stocks overview
    st.markdown("### 📊 Popular Indian Stocks")
    display_popular_stocks_overview()

def display_popular_stocks_overview():
    """Display overview of popular Indian stocks"""
    
    # Select a few popular stocks for overview
    overview_stocks = POPULAR_STOCKS[:10]
    
    with st.spinner("Fetching market data..."):
        stock_overview = []
        
        for symbol in overview_stocks:
            try:
                # Get basic info
                info = stock_fetcher.fetch_stock_info(symbol)
                current_price = stock_fetcher.get_real_time_price(symbol)
                
                if info and current_price:
                    stock_overview.append({
                        'Symbol': symbol,
                        'Company': info.get('company_name', symbol)[:30],
                        'Price (₹)': f"{current_price:.2f}",
                        'Market Cap': format_currency(info.get('market_cap', 0)),
                        'P/E Ratio': f"{info.get('pe_ratio', 0):.2f}",
                        'Sector': info.get('sector', 'N/A')
                    })
            except:
                continue
    
    if stock_overview:
        df = pd.DataFrame(stock_overview)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Unable to fetch market overview at this time")

def display_market_overview():
    """Display market indices overview"""
    
    st.markdown("### 📈 Market Indices Overview")
    
    with st.spinner("Fetching market indices..."):
        indices_data = stock_fetcher.fetch_market_indices()
    
    if indices_data:
        cols = st.columns(len(indices_data))
        
        for i, (name, data) in enumerate(indices_data.items()):
            with cols[i]:
                current_value = data['Close'].iloc[-1]
                prev_value = data['Close'].iloc[-2]
                change = current_value - prev_value
                change_pct = (change / prev_value) * 100
                
                color = "positive" if change >= 0 else "negative"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <h3>₹{current_value:,.2f}</h3>
                    <p class="{color}">{change:+.2f} ({change_pct:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)

def display_stock_analysis():
    """Display comprehensive stock analysis"""
    
    symbol = st.session_state.selected_stock
    exchange = st.session_state.exchange
    period = st.session_state.period
    
    st.markdown(f"## 📊 Analysis for {symbol} ({exchange})")
    
    # Fetch stock data
    with st.spinner("Fetching stock data..."):
        stock_data = stock_fetcher.fetch_stock_data(symbol, period, exchange)
        stock_info = stock_fetcher.fetch_stock_info(symbol, exchange)
    
    if stock_data is None or stock_data.empty:
        st.error("Unable to fetch stock data. Please check the symbol and try again.")
        return
    
    # Display basic info
    display_stock_overview(stock_data, stock_info, symbol)
    
    # Create tabs for different analysis
    tabs = []
    if st.session_state.get('show_technical', True):
        tabs.append("Technical Analysis")
    if st.session_state.get('show_fundamental', True):
        tabs.append("Fundamental Analysis")
    if st.session_state.get('show_sentiment', True):
        tabs.append("Sentiment Analysis")
    if st.session_state.get('show_rl', True):
        tabs.append("RL Predictions")
    
    if tabs:
        tab_objects = st.tabs(tabs)
        
        tab_index = 0
        
        if st.session_state.get('show_technical', True):
            with tab_objects[tab_index]:
                display_technical_analysis(stock_data, symbol)
            tab_index += 1
        
        if st.session_state.get('show_fundamental', True):
            with tab_objects[tab_index]:
                display_fundamental_analysis(symbol, exchange)
            tab_index += 1
        
        if st.session_state.get('show_sentiment', True):
            with tab_objects[tab_index]:
                display_sentiment_analysis(symbol)
            tab_index += 1
        
        if st.session_state.get('show_rl', True):
            with tab_objects[tab_index]:
                display_rl_analysis(stock_data, symbol)

def display_stock_overview(stock_data, stock_info, symbol):
    """Display basic stock information and price chart"""
    
    current_price = stock_data['Close'].iloc[-1]
    prev_price = stock_data['Close'].iloc[-2]
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "positive" if change >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Price</h4>
            <h3>₹{current_price:.2f}</h3>
            <p class="{color}">{change:+.2f} ({change_pct:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock_data['Volume'].rolling(30).mean().iloc[-1]
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volume</h4>
            <h3>{volume:,.0f}</h3>
            <p>Avg: {avg_volume:,.0f} ({volume_ratio:.1f}x)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_52w = stock_info.get('fifty_two_week_high', 0) if stock_info else 0
        low_52w = stock_info.get('fifty_two_week_low', 0) if stock_info else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>52W High</h4>
            <h3>₹{high_52w:.2f}</h3>
            <p>Low: ₹{low_52w:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        market_cap = stock_info.get('market_cap', 0) if stock_info else 0
        pe_ratio = stock_info.get('pe_ratio', 0) if stock_info else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Cap</h4>
            <h3>{format_currency(market_cap)}</h3>
            <p>P/E: {pe_ratio:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Price chart
    st.subheader("📈 Price Chart")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price', 'Volume'),
        vertical_spacing=0.1,
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_analysis(stock_data, symbol):
    """Display technical analysis"""
    
    st.subheader("🔍 Technical Analysis")
    
    with st.spinner("Calculating technical indicators..."):
        # Calculate indicators
        rsi = technical_analyzer.calculate_rsi(stock_data)
        macd = technical_analyzer.calculate_macd(stock_data)
        ma = technical_analyzer.calculate_moving_averages(stock_data)
        bb = technical_analyzer.calculate_bollinger_bands(stock_data)
        stoch = technical_analyzer.calculate_stochastic(stock_data)
        signals = technical_analyzer.generate_signals(stock_data)
        summary = technical_analyzer.get_technical_summary(stock_data)
    
    # Technical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Current Indicators")
        
        # RSI
        rsi_color = "negative" if summary['rsi'] > 70 else "positive" if summary['rsi'] < 30 else "neutral"
        st.markdown(f"**RSI**: <span class='{rsi_color}'>{summary['rsi']:.2f}</span> - {summary['rsi_interpretation']}", unsafe_allow_html=True)
        
        # MACD
        st.markdown(f"**MACD**: {summary['macd_interpretation']}")
        
        # Moving Averages
        st.markdown(f"**Moving Averages**: {summary['ma_interpretation']}")
        
        # Bollinger Bands
        st.markdown(f"**Bollinger Bands**: {summary['bb_position']}")
    
    with col2:
        st.markdown("### 🎯 Trading Signals")
        
        # Recent signals
        recent_signals = signals.tail(5)
        for date, row in recent_signals.iterrows():
            if row['Buy_Signal']:
                st.success(f"BUY signal on {date.strftime('%Y-%m-%d')}")
            elif row['Sell_Signal']:
                st.error(f"SELL signal on {date.strftime('%Y-%m-%d')}")
    
    # Technical charts
    create_technical_charts(stock_data, rsi, macd, ma, bb, stoch)

def create_technical_charts(stock_data, rsi, macd, ma, bb, stoch):
    """Create technical analysis charts"""
    
    # RSI Chart
    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='blue')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title="RSI Indicator", yaxis_title="RSI", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD Chart
    st.subheader("MACD")
    fig_macd = make_subplots(rows=2, cols=1, subplot_titles=('MACD Line & Signal', 'MACD Histogram'))
    
    fig_macd.add_trace(go.Scatter(x=macd.index, y=macd['MACD'], name='MACD', line=dict(color='blue')), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=macd.index, y=macd['Signal'], name='Signal', line=dict(color='red')), row=1, col=1)
    fig_macd.add_trace(go.Bar(x=macd.index, y=macd['Histogram'], name='Histogram'), row=2, col=1)
    
    fig_macd.update_layout(height=400)
    st.plotly_chart(fig_macd, use_container_width=True)
    
    # Price with Bollinger Bands and Moving Averages
    st.subheader("Price with Technical Indicators")
    fig_price = go.Figure()
    
    # Price
    fig_price.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price', line=dict(color='black')))
    
    # Bollinger Bands
    fig_price.add_trace(go.Scatter(x=bb.index, y=bb['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')))
    fig_price.add_trace(go.Scatter(x=bb.index, y=bb['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')))
    fig_price.add_trace(go.Scatter(x=bb.index, y=bb['BB_Middle'], name='BB Middle', line=dict(color='orange')))
    
    # Moving Averages
    fig_price.add_trace(go.Scatter(x=ma.index, y=ma['SMA_20'], name='SMA 20', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(x=ma.index, y=ma['SMA_50'], name='SMA 50', line=dict(color='red')))
    
    fig_price.update_layout(title="Price with Technical Indicators", height=500)
    st.plotly_chart(fig_price, use_container_width=True)

def display_fundamental_analysis(symbol, exchange):
    """Display fundamental analysis"""
    
    st.subheader("📊 Fundamental Analysis")
    
    with st.spinner("Analyzing fundamentals..."):
        analysis = fundamental_analyzer.analyze_fundamentals(symbol, exchange)
    
    if not analysis:
        st.error("Unable to fetch fundamental data")
        return
    
    # Overall recommendation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rec_color = "positive" if "Buy" in analysis['recommendation'] else "negative" if "Sell" in analysis['recommendation'] else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recommendation</h4>
            <h3 class="{rec_color}">{analysis['recommendation']}</h3>
            <p>Score: {analysis['overall_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk Level</h4>
            <h3>{analysis['risk_level']}</h3>
            <p>Growth: {analysis['growth_potential']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        financial_strength_color = "positive" if analysis.get('financial_strength') == "Strong" else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Financial Strength</h4>
            <h3 class="{financial_strength_color}">{analysis.get('financial_strength', 'N/A')}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed scores
    st.subheader("📈 Detailed Scores")
    
    scores = analysis['scores']
    score_df = pd.DataFrame([
        {'Metric': 'Valuation', 'Score': scores.get('valuation', 0)},
        {'Metric': 'Profitability', 'Score': scores.get('profitability', 0)},
        {'Metric': 'Growth', 'Score': scores.get('growth', 0)},
        {'Metric': 'Financial Health', 'Score': scores.get('financial_health', 0)}
    ])
    
    # Score visualization
    fig_scores = px.bar(score_df, x='Metric', y='Score', title="Fundamental Analysis Scores")
    fig_scores.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_scores, use_container_width=True)

def display_sentiment_analysis(symbol):
    """Display sentiment analysis"""
    
    st.subheader("🎭 Sentiment Analysis")
    
    with st.spinner("Analyzing market sentiment..."):
        sentiment = sentiment_analyzer.analyze_stock_sentiment(symbol)
    
    # Sentiment overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_color = "positive" if sentiment['overall_sentiment'] == "Bullish" else "negative" if sentiment['overall_sentiment'] == "Bearish" else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Overall Sentiment</h4>
            <h3 class="{sentiment_color}">{sentiment['overall_sentiment']}</h3>
            <p>Score: {sentiment['sentiment_score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sentiment Strength</h4>
            <h3>{sentiment['sentiment_strength']}</h3>
            <p>Consistency: {sentiment['consistency']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>News Articles</h4>
            <h3>{sentiment['total_articles']}</h3>
            <p>Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    
    sentiment_data = {
        'Category': ['Positive', 'Negative', 'Neutral'],
        'Percentage': [
            sentiment['positive_percentage'],
            sentiment['negative_percentage'],
            sentiment['neutral_percentage']
        ]
    }
    
    fig_sentiment = px.pie(
        values=sentiment_data['Percentage'],
        names=sentiment_data['Category'],
        title="News Sentiment Distribution",
        color_discrete_map={
            'Positive': '#00C851',
            'Negative': '#ff4444',
            'Neutral': '#ffbb33'
        }
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)

def display_rl_analysis(stock_data, symbol):
    """Display RL model analysis and predictions"""
    
    st.subheader("🤖 Reinforcement Learning Analysis")
    
    # Check if model is trained
    if not rl_trading_system.trained:
        # Try to load existing model
        if rl_trading_system.load_model():
            st.success("✅ Pre-trained model loaded successfully!")
        else:
            st.warning("⚠️ No trained model found. Please train the model first.")
            if st.button("Train Model Now"):
                train_rl_model()
            return
    
    # Get RL prediction
    with st.spinner("Getting RL predictions..."):
        action, confidence = rl_trading_system.predict_action(stock_data, symbol)
    
    # Display prediction
    col1, col2 = st.columns(2)
    
    with col1:
        action_color = "positive" if action == "BUY" else "negative" if action == "SELL" else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <h4>RL Recommendation</h4>
            <h3 class="{action_color}">{action}</h3>
            <p>Confidence: {confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Confidence Level</h4>
            <h3>{confidence_level}</h3>
            <p>{confidence*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Backtesting results
    st.subheader("📈 Strategy Backtesting")
    
    with st.spinner("Running backtest..."):
        backtest_results = rl_trading_system.backtest_strategy(stock_data, symbol)
    
    if backtest_results:
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = backtest_results['total_return']
            return_color = "positive" if total_return > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Return</h4>
                <h3 class="{return_color}">{total_return*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            benchmark_return = backtest_results['benchmark_return']
            bench_color = "positive" if benchmark_return > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Benchmark Return</h4>
                <h3 class="{bench_color}">{benchmark_return*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            excess_return = backtest_results['excess_return']
            excess_color = "positive" if excess_return > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Excess Return</h4>
                <h3 class="{excess_color}">{excess_return*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sharpe_ratio = backtest_results['sharpe_ratio']
            sharpe_color = "positive" if sharpe_ratio > 1 else "neutral" if sharpe_ratio > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Sharpe Ratio</h4>
                <h3 class="{sharpe_color}">{sharpe_ratio:.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Portfolio value chart
        portfolio_df = pd.DataFrame({
            'Portfolio Value': backtest_results['portfolio_values']
        }, index=stock_data.index[-len(backtest_results['portfolio_values']):])
        
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['Portfolio Value'],
            name='RL Strategy',
            line=dict(color='blue')
        ))
        
        # Add benchmark
        benchmark_data = stock_data['Close'][-len(backtest_results['portfolio_values']):]
        benchmark_portfolio = (benchmark_data / benchmark_data.iloc[0]) * 100000  # Normalize to same starting value
        
        fig_portfolio.add_trace(go.Scatter(
            x=benchmark_data.index,
            y=benchmark_portfolio,
            name='Buy & Hold',
            line=dict(color='red', dash='dash')
        ))
        
        fig_portfolio.update_layout(
            title="Portfolio Performance Comparison",
            yaxis_title="Portfolio Value (₹)",
            height=400
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Trading summary
        st.subheader("💼 Trading Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Trades", backtest_results['total_trades'])
            st.metric("Transaction Costs", format_currency(backtest_results['transaction_costs']))
        
        with col2:
            st.metric("Max Drawdown", f"{backtest_results['max_drawdown']*100:.2f}%")

def train_rl_model():
    """Train the RL model"""
    
    symbol = st.session_state.selected_stock
    exchange = st.session_state.exchange
    period = "2y"  # Use longer period for training
    
    st.info("🤖 Training RL model... This may take a few minutes.")
    
    # Fetch training data
    with st.spinner("Fetching training data..."):
        training_data = stock_fetcher.fetch_stock_data(symbol, period, exchange)
    
    if training_data is None or len(training_data) < 100:
        st.error("Insufficient data for training. Please try a different stock or period.")
        return
    
    # Train model
    try:
        with st.spinner("Training model..."):
            episode_rewards = rl_trading_system.train_model(training_data, symbol, episodes=50)
        
        st.success("✅ Model trained successfully!")
        
        # Display training progress
        if episode_rewards:
            fig_training = go.Figure()
            fig_training.add_trace(go.Scatter(
                y=episode_rewards,
                mode='lines',
                name='Episode Rewards'
            ))
            fig_training.update_layout(
                title="Training Progress",
                xaxis_title="Episode",
                yaxis_title="Reward",
                height=300
            )
            st.plotly_chart(fig_training, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    # Initialize session state
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = ""
    if 'exchange' not in st.session_state:
        st.session_state.exchange = "NSE"
    if 'period' not in st.session_state:
        st.session_state.period = "1y"
    
    # Run main application
    main()
