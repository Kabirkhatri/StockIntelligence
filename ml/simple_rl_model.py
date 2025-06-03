"""
Simplified RL model for stock trading without TensorFlow dependencies
"""
import numpy as np
import pandas as pd
import streamlit as st
from config.settings import INITIAL_CAPITAL, RL_ACTIONS
import random

class SimpleTradingAgent:
    """Simple trading agent using basic Q-learning"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        
    def get_state_key(self, state_features):
        """Convert state features to a hashable key"""
        # Discretize continuous features for Q-table
        rsi_bucket = int(state_features.get('rsi', 50) / 10)
        macd_bucket = 1 if state_features.get('macd', 0) > 0 else 0
        ma_bucket = 1 if state_features.get('price_above_ma', False) else 0
        
        return f"{rsi_bucket}_{macd_bucket}_{ma_bucket}"
    
    def choose_action(self, state_features):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(RL_ACTIONS)
        
        if random.random() < self.epsilon:
            return random.choice(RL_ACTIONS)
        
        best_action_idx = np.argmax(self.q_table[state_key])
        return RL_ACTIONS[best_action_idx]
    
    def update_q_value(self, state_features, action, reward, next_state_features):
        """Update Q-value based on experience"""
        state_key = self.get_state_key(state_features)
        next_state_key = self.get_state_key(next_state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(RL_ACTIONS)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * len(RL_ACTIONS)
        
        action_idx = RL_ACTIONS.index(action)
        current_q = self.q_table[state_key][action_idx]
        max_next_q = max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

class SimpleRLTradingSystem:
    """Simple RL trading system"""
    
    def __init__(self):
        self.agent = SimpleTradingAgent()
        self.trained = False
    
    def extract_features(self, data, current_idx):
        """Extract features from stock data"""
        if current_idx < 20:
            return {
                'rsi': 50,
                'macd': 0,
                'price_above_ma': True
            }
        
        current_price = data['Close'].iloc[current_idx]
        recent_data = data.iloc[max(0, current_idx-20):current_idx+1]
        
        # Simple RSI calculation
        price_changes = recent_data['Close'].diff()
        gains = price_changes.where(price_changes > 0, 0).mean()
        losses = -price_changes.where(price_changes < 0, 0).mean()
        
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        
        # Simple MACD
        ema_12 = recent_data['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = recent_data['Close'].ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        
        # Moving average
        ma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
        price_above_ma = current_price > ma_20
        
        return {
            'rsi': rsi,
            'macd': macd,
            'price_above_ma': price_above_ma
        }
    
    def train_model(self, stock_data, symbol, episodes=50):
        """Train the simple RL model"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for episode in range(episodes):
            self.simulate_trading(stock_data, training=True)
            
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            status_text.text(f"Training Episode {episode + 1}/{episodes}")
        
        progress_bar.empty()
        status_text.empty()
        self.trained = True
        
        return list(range(episodes))  # Dummy rewards for compatibility
    
    def simulate_trading(self, data, training=False):
        """Simulate trading with the RL agent"""
        cash = INITIAL_CAPITAL
        shares = 0
        
        for i in range(20, len(data) - 1):
            state_features = self.extract_features(data, i)
            action = self.agent.choose_action(state_features)
            
            current_price = data['Close'].iloc[i]
            next_price = data['Close'].iloc[i + 1]
            
            # Execute action
            prev_value = cash + shares * current_price
            
            if action == 'BUY' and cash > current_price:
                shares_to_buy = int(cash * 0.9 / current_price)
                cash -= shares_to_buy * current_price
                shares += shares_to_buy
            elif action == 'SELL' and shares > 0:
                cash += shares * current_price
                shares = 0
            
            new_value = cash + shares * next_price
            reward = (new_value - prev_value) / prev_value
            
            if training:
                next_state_features = self.extract_features(data, i + 1)
                self.agent.update_q_value(state_features, action, reward, next_state_features)
    
    def predict_action(self, stock_data, symbol):
        """Predict trading action"""
        if len(stock_data) < 20:
            return "HOLD", 0.5
        
        features = self.extract_features(stock_data, len(stock_data) - 1)
        action = self.agent.choose_action(features)
        
        # Simple confidence based on Q-values
        state_key = self.agent.get_state_key(features)
        if state_key in self.agent.q_table:
            q_values = self.agent.q_table[state_key]
            confidence = (max(q_values) - min(q_values)) / (max(q_values) + 1e-8)
        else:
            confidence = 0.5
        
        return action, min(max(confidence, 0), 1)
    
    def backtest_strategy(self, stock_data, symbol):
        """Backtest the strategy"""
        if not self.trained:
            return None
        
        portfolio_values = [INITIAL_CAPITAL]
        actions_taken = []
        cash = INITIAL_CAPITAL
        shares = 0
        
        for i in range(20, len(stock_data)):
            features = self.extract_features(stock_data, i)
            action = self.agent.choose_action(features)
            
            current_price = stock_data['Close'].iloc[i]
            
            if action == 'BUY' and cash > current_price:
                shares_to_buy = int(cash * 0.9 / current_price)
                cash -= shares_to_buy * current_price
                shares += shares_to_buy
            elif action == 'SELL' and shares > 0:
                cash += shares * current_price
                shares = 0
            
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)
            actions_taken.append(action)
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        return {
            'total_return': total_return,
            'final_value': portfolio_values[-1],
            'portfolio_values': portfolio_values,
            'actions': actions_taken,
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_drawdown': min(returns.cumsum())
        }

# Create global instance
simple_rl_system = SimpleRLTradingSystem()