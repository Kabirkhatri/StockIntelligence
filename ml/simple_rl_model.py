
"""
Simplified RL model for stock trading without TensorFlow dependencies
"""
import numpy as np
import pandas as pd
import streamlit as st
from config.settings import INITIAL_CAPITAL, RL_ACTIONS
import random
import pickle
import os

class SimpleTradingAgent:
    """Simple trading agent using basic Q-learning"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.3  # Higher exploration initially
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
    def get_state_key(self, state_features):
        """Convert state features to a hashable key"""
        try:
            # Discretize continuous features for Q-table
            rsi = state_features.get('rsi', 50)
            macd = state_features.get('macd', 0)
            price_trend = state_features.get('price_trend', 0)
            volume_ratio = state_features.get('volume_ratio', 1)
            
            rsi_bucket = min(max(int(rsi / 20), 0), 4)  # 0-4 buckets
            macd_bucket = 2 if macd > 0.01 else 1 if macd > -0.01 else 0
            trend_bucket = 2 if price_trend > 0.02 else 1 if price_trend > -0.02 else 0
            vol_bucket = min(max(int(volume_ratio), 0), 3)  # 0-3 buckets
            
            return f"{rsi_bucket}_{macd_bucket}_{trend_bucket}_{vol_bucket}"
        except:
            return "default_state"
    
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
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the Q-table"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon
                }, f)
            return True
        except:
            return False
    
    def load_model(self, filepath):
        """Load the Q-table"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    self.epsilon = data.get('epsilon', 0.01)
                return True
        except:
            pass
        return False

class SimpleRLTradingSystem:
    """Simple RL trading system"""
    
    def __init__(self):
        self.agent = SimpleTradingAgent()
        self.trained = False
        self.model_path = "simple_rl_model.pkl"
        
        # Try to load existing model
        if self.agent.load_model(self.model_path):
            self.trained = True
    
    def extract_features(self, data, current_idx):
        """Extract features from stock data"""
        try:
            if current_idx < 20:
                return {
                    'rsi': 50,
                    'macd': 0,
                    'price_trend': 0,
                    'volume_ratio': 1
                }
            
            current_price = data['Close'].iloc[current_idx]
            recent_data = data.iloc[max(0, current_idx-20):current_idx+1]
            
            # Ensure we have enough data
            if len(recent_data) < 10:
                return {
                    'rsi': 50,
                    'macd': 0,
                    'price_trend': 0,
                    'volume_ratio': 1
                }
            
            # Simple RSI calculation
            price_changes = recent_data['Close'].diff().dropna()
            if len(price_changes) == 0:
                rsi = 50
            else:
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                
                avg_gain = gains.rolling(min(14, len(gains))).mean().iloc[-1]
                avg_loss = losses.rolling(min(14, len(losses))).mean().iloc[-1]
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
            
            # Simple MACD
            try:
                if len(recent_data) >= 12:
                    ema_12 = recent_data['Close'].ewm(span=12).mean().iloc[-1]
                    ema_26 = recent_data['Close'].ewm(span=min(26, len(recent_data))).mean().iloc[-1]
                    macd = (ema_12 - ema_26) / ema_26  # Normalize by price
                else:
                    macd = 0
            except:
                macd = 0
            
            # Price trend (5-day vs 10-day moving average)
            try:
                if len(recent_data) >= 10:
                    ma_5 = recent_data['Close'].rolling(5).mean().iloc[-1]
                    ma_10 = recent_data['Close'].rolling(10).mean().iloc[-1]
                    price_trend = (ma_5 - ma_10) / ma_10  # Normalized trend
                else:
                    price_trend = 0
            except:
                price_trend = 0
            
            # Volume ratio
            try:
                current_volume = data['Volume'].iloc[current_idx]
                avg_volume = recent_data['Volume'].rolling(10).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            except:
                volume_ratio = 1
            
            return {
                'rsi': rsi,
                'macd': macd,
                'price_trend': price_trend,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            # Return default features on error
            return {
                'rsi': 50,
                'macd': 0,
                'price_trend': 0,
                'volume_ratio': 1
            }
    
    def train_model(self, stock_data, symbol, episodes=100):
        """Train the simple RL model"""
        if len(stock_data) < 50:
            st.error("Insufficient data for training. Need at least 50 data points.")
            return []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        episode_rewards = []
        
        for episode in range(episodes):
            total_reward = self.simulate_trading(stock_data, training=True)
            episode_rewards.append(total_reward)
            
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            status_text.text(f"Training Episode {episode + 1}/{episodes} - Reward: {total_reward:.2f}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Save trained model
        self.agent.save_model(self.model_path)
        self.trained = True
        
        return episode_rewards
    
    def simulate_trading(self, data, training=False):
        """Simulate trading with the RL agent"""
        cash = INITIAL_CAPITAL
        shares = 0
        total_reward = 0
        max_position_value = INITIAL_CAPITAL * 0.8  # Max 80% of portfolio in stocks
        
        # Start from a safe index
        start_idx = 25
        end_idx = len(data) - 1
        
        for i in range(start_idx, end_idx):
            try:
                state_features = self.extract_features(data, i)
                
                # Choose action
                if training:
                    action = self.agent.choose_action(state_features)
                else:
                    # Use exploitation during backtesting
                    old_epsilon = self.agent.epsilon
                    self.agent.epsilon = 0
                    action = self.agent.choose_action(state_features)
                    self.agent.epsilon = old_epsilon
                
                current_price = data['Close'].iloc[i]
                next_price = data['Close'].iloc[i + 1]
                
                # Calculate portfolio value before action
                prev_value = cash + shares * current_price
                
                # Execute action with realistic position sizing
                if action == 'BUY' and cash > current_price * 10:  # Minimum 10 shares
                    # Calculate maximum position value
                    current_position_value = shares * current_price
                    available_for_stocks = max_position_value - current_position_value
                    
                    if available_for_stocks > 0:
                        # Use only 20% of available cash per trade for gradual position building
                        max_investment = min(cash * 0.2, available_for_stocks)
                        shares_to_buy = int(max_investment / current_price)
                        
                        # Cap at reasonable trade size (max 1% of typical daily volume)
                        max_trade_shares = max(10, int(data['Volume'].iloc[i] * 0.001))
                        shares_to_buy = min(shares_to_buy, max_trade_shares)
                        
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price
                            # Add transaction cost (0.1%)
                            transaction_cost = cost * 0.001
                            total_cost = cost + transaction_cost
                            
                            if cash >= total_cost:
                                cash -= total_cost
                                shares += shares_to_buy
                        
                elif action == 'SELL' and shares > 0:
                    # Sell only 50% of position for gradual exit
                    shares_to_sell = max(1, shares // 2)
                    revenue = shares_to_sell * current_price
                    # Subtract transaction cost (0.1%)
                    transaction_cost = revenue * 0.001
                    net_revenue = revenue - transaction_cost
                    
                    cash += net_revenue
                    shares -= shares_to_sell
                
                # Calculate new portfolio value and reward
                new_value = cash + shares * next_price
                
                # More conservative reward calculation
                if prev_value > 0:
                    # Portfolio return
                    portfolio_return = (new_value - prev_value) / prev_value
                    
                    # Market return (benchmark)
                    market_return = (next_price - current_price) / current_price
                    
                    # Reward based on excess return over market, scaled down
                    reward = (portfolio_return - market_return) * 50  # Reduced scaling
                    
                    # Conservative reward shaping
                    if action == 'BUY' and next_price > current_price * 1.005:  # Only reward significant gains
                        reward += 0.5
                    elif action == 'SELL' and next_price < current_price * 0.995:  # Only reward avoiding significant losses
                        reward += 0.5
                    elif action == 'HOLD':
                        reward += 0.05  # Small reward for holding
                    
                    # Penalty for poor timing
                    if action == 'BUY' and next_price < current_price * 0.99:
                        reward -= 1
                    elif action == 'SELL' and next_price > current_price * 1.01:
                        reward -= 1
                        
                else:
                    reward = 0
                
                total_reward += reward
                
                # Update Q-values during training
                if training and i < end_idx - 1:
                    next_state_features = self.extract_features(data, i + 1)
                    self.agent.update_q_value(state_features, action, reward, next_state_features)
                    
            except Exception as e:
                continue
        
        return total_reward
    
    def predict_action(self, stock_data, symbol):
        """Predict trading action"""
        try:
            if len(stock_data) < 25:
                return "HOLD", 0.5
            
            # Load model if not trained
            if not self.trained:
                if self.agent.load_model(self.model_path):
                    self.trained = True
                else:
                    return "HOLD", 0.5
            
            features = self.extract_features(stock_data, len(stock_data) - 1)
            
            # Get action with no exploration
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            action = self.agent.choose_action(features)
            self.agent.epsilon = old_epsilon
            
            # Calculate confidence based on Q-values
            state_key = self.agent.get_state_key(features)
            if state_key in self.agent.q_table:
                q_values = self.agent.q_table[state_key]
                if max(q_values) != min(q_values):
                    # Normalize confidence between 0.3 and 0.9
                    raw_confidence = (max(q_values) - np.mean(q_values)) / (max(q_values) - min(q_values))
                    confidence = 0.3 + (raw_confidence * 0.6)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            return action, min(max(confidence, 0), 1)
            
        except Exception as e:
            return "HOLD", 0.5
    
    def backtest_strategy(self, stock_data, symbol):
        """Backtest the strategy"""
        try:
            if not self.trained:
                if not self.agent.load_model(self.model_path):
                    return None
                self.trained = True
            
            if len(stock_data) < 50:
                return None
            
            portfolio_values = [INITIAL_CAPITAL]
            actions_taken = []
            trades = []
            cash = INITIAL_CAPITAL
            shares = 0
            
            start_idx = 25
            
            max_position_value = INITIAL_CAPITAL * 0.8  # Max 80% of portfolio in stocks
            
            for i in range(start_idx, len(stock_data)):
                try:
                    features = self.extract_features(stock_data, i)
                    
                    # Use trained policy (no exploration)
                    old_epsilon = self.agent.epsilon
                    self.agent.epsilon = 0
                    action = self.agent.choose_action(features)
                    self.agent.epsilon = old_epsilon
                    
                    current_price = stock_data['Close'].iloc[i]
                    
                    # Execute action with realistic trading logic
                    if action == 'BUY' and cash > current_price * 10:  # Minimum 10 shares
                        current_position_value = shares * current_price
                        available_for_stocks = max_position_value - current_position_value
                        
                        if available_for_stocks > 0:
                            # Use only 20% of available cash per trade
                            max_investment = min(cash * 0.2, available_for_stocks)
                            shares_to_buy = int(max_investment / current_price)
                            
                            # Cap at reasonable trade size
                            max_trade_shares = max(10, int(stock_data['Volume'].iloc[i] * 0.001))
                            shares_to_buy = min(shares_to_buy, max_trade_shares)
                            
                            if shares_to_buy > 0:
                                cost = shares_to_buy * current_price
                                transaction_cost = cost * 0.001  # 0.1% transaction cost
                                total_cost = cost + transaction_cost
                                
                                if cash >= total_cost:
                                    cash -= total_cost
                                    shares += shares_to_buy
                                    trades.append({
                                        'action': 'BUY',
                                        'shares': shares_to_buy,
                                        'price': current_price,
                                        'date': stock_data.index[i] if hasattr(stock_data.index[i], 'strftime') else i
                                    })
                            
                    elif action == 'SELL' and shares > 0:
                        # Sell only 50% of position for gradual exit
                        shares_to_sell = max(1, shares // 2)
                        revenue = shares_to_sell * current_price
                        transaction_cost = revenue * 0.001  # 0.1% transaction cost
                        net_revenue = revenue - transaction_cost
                        
                        cash += net_revenue
                        shares -= shares_to_sell
                        trades.append({
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'price': current_price,
                            'date': stock_data.index[i] if hasattr(stock_data.index[i], 'strftime') else i
                        })
                    
                    portfolio_value = cash + shares * current_price
                    portfolio_values.append(portfolio_value)
                    actions_taken.append(action)
                    
                except:
                    # On error, just hold
                    portfolio_value = cash + shares * stock_data['Close'].iloc[i]
                    portfolio_values.append(portfolio_value)
                    actions_taken.append('HOLD')
            
            # Calculate performance metrics
            if len(portfolio_values) > 1:
                returns = pd.Series(portfolio_values).pct_change().dropna()
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                
                # Calculate Sharpe ratio
                if returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                
                # Calculate max drawdown
                peak = pd.Series(portfolio_values).expanding().max()
                drawdown = (pd.Series(portfolio_values) - peak) / peak
                max_drawdown = drawdown.min()
                
                return {
                    'total_return': total_return,
                    'final_value': portfolio_values[-1],
                    'portfolio_values': portfolio_values,
                    'actions': actions_taken,
                    'trades': trades,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
            
        except Exception as e:
            st.error(f"Error in backtesting: {str(e)}")
            return None
        
        return None

# Create global instance
simple_rl_system = SimpleRLTradingSystem()
