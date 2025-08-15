
"""
Simplified RL model for stock trading without TensorFlow dependencies
Enhanced for better profitability and reduced gap with benchmark
"""
import numpy as np
import pandas as pd
import streamlit as st
from config.settings import INITIAL_CAPITAL, RL_ACTIONS
import random
import pickle
import os

class SimpleTradingAgent:
    """Simple trading agent using basic Q-learning with enhanced trading logic"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.15  # Increased learning rate
        self.discount_factor = 0.98  # Higher discount for long-term rewards
        self.epsilon = 0.4  # Higher initial exploration
        self.epsilon_decay = 0.998  # Slower decay to maintain exploration
        self.min_epsilon = 0.05
        
        # Enhanced state tracking
        self.state_visits = {}
        self.action_rewards = {'BUY': [], 'SELL': [], 'HOLD': []}
        
    def get_state_key(self, state_features):
        """Convert state features to a hashable key with enhanced discretization"""
        try:
            # More granular bucketing for better decision making
            rsi = state_features.get('rsi', 50)
            macd = state_features.get('macd', 0)
            price_trend = state_features.get('price_trend', 0)
            volume_ratio = state_features.get('volume_ratio', 1)
            momentum = state_features.get('momentum', 0)
            volatility = state_features.get('volatility', 0)
            
            # Enhanced bucketing
            rsi_bucket = min(max(int(rsi / 10), 0), 9)  # 10 buckets (0-9)
            macd_bucket = 4 if macd > 0.02 else 3 if macd > 0.01 else 2 if macd > -0.01 else 1 if macd > -0.02 else 0
            trend_bucket = 4 if price_trend > 0.03 else 3 if price_trend > 0.01 else 2 if price_trend > -0.01 else 1 if price_trend > -0.03 else 0
            vol_bucket = min(max(int(volume_ratio * 2), 0), 5)  # 6 buckets
            mom_bucket = 2 if momentum > 0.02 else 1 if momentum > -0.02 else 0
            volatility_bucket = min(max(int(volatility * 100), 0), 3)  # 4 buckets
            
            return f"{rsi_bucket}_{macd_bucket}_{trend_bucket}_{vol_bucket}_{mom_bucket}_{volatility_bucket}"
        except:
            return "default_state"
    
    def choose_action(self, state_features):
        """Choose action using epsilon-greedy policy with enhanced logic"""
        state_key = self.get_state_key(state_features)
        
        if state_key not in self.q_table:
            # Initialize with slight bias towards profitable actions
            self.q_table[state_key] = [0.1, 0.1, 0.0]  # BUY, SELL, HOLD
        
        # Track state visits for adaptive learning
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
        
        # Adaptive epsilon based on state familiarity
        adaptive_epsilon = self.epsilon / (1 + self.state_visits[state_key] * 0.1)
        
        if random.random() < adaptive_epsilon:
            # Smart exploration - bias towards potentially profitable actions
            rsi = state_features.get('rsi', 50)
            trend = state_features.get('price_trend', 0)
            
            if rsi < 30 and trend > 0:  # Oversold with positive trend
                return 'BUY' if random.random() < 0.7 else random.choice(RL_ACTIONS)
            elif rsi > 70 and trend < 0:  # Overbought with negative trend
                return 'SELL' if random.random() < 0.7 else random.choice(RL_ACTIONS)
            else:
                return random.choice(RL_ACTIONS)
        
        best_action_idx = np.argmax(self.q_table[state_key])
        return RL_ACTIONS[best_action_idx]
    
    def update_q_value(self, state_features, action, reward, next_state_features):
        """Update Q-value with enhanced learning"""
        state_key = self.get_state_key(state_features)
        next_state_key = self.get_state_key(next_state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(RL_ACTIONS)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * len(RL_ACTIONS)
        
        action_idx = RL_ACTIONS.index(action)
        current_q = self.q_table[state_key][action_idx]
        max_next_q = max(self.q_table[next_state_key])
        
        # Enhanced Q-learning with momentum
        learning_rate = self.learning_rate / (1 + self.state_visits.get(state_key, 0) * 0.01)
        new_q = current_q + learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
        
        # Track action performance
        self.action_rewards[action].append(reward)
        if len(self.action_rewards[action]) > 100:
            self.action_rewards[action].pop(0)
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the Q-table and learning state"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'state_visits': self.state_visits,
                    'action_rewards': self.action_rewards
                }, f)
            return True
        except:
            return False
    
    def load_model(self, filepath):
        """Load the Q-table and learning state"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    self.epsilon = data.get('epsilon', 0.05)
                    self.state_visits = data.get('state_visits', {})
                    self.action_rewards = data.get('action_rewards', {'BUY': [], 'SELL': [], 'HOLD': []})
                return True
        except:
            pass
        return False

class SimpleRLTradingSystem:
    """Enhanced RL trading system with improved profitability"""
    
    def __init__(self):
        self.agent = SimpleTradingAgent()
        self.trained = False
        self.model_path = "simple_rl_model.pkl"
        
        # Try to load existing model
        if self.agent.load_model(self.model_path):
            self.trained = True
    
    def extract_features(self, data, current_idx):
        """Extract enhanced features from stock data"""
        try:
            if current_idx < 25:
                return {
                    'rsi': 50, 'macd': 0, 'price_trend': 0, 'volume_ratio': 1,
                    'momentum': 0, 'volatility': 0
                }
            
            current_price = data['Close'].iloc[current_idx]
            recent_data = data.iloc[max(0, current_idx-25):current_idx+1]
            
            if len(recent_data) < 15:
                return {
                    'rsi': 50, 'macd': 0, 'price_trend': 0, 'volume_ratio': 1,
                    'momentum': 0, 'volatility': 0
                }
            
            # Enhanced RSI calculation
            price_changes = recent_data['Close'].diff().dropna()
            if len(price_changes) == 0:
                rsi = 50
            else:
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                
                period = min(14, len(gains))
                avg_gain = gains.rolling(period).mean().iloc[-1]
                avg_loss = losses.rolling(period).mean().iloc[-1]
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
            
            # Enhanced MACD
            try:
                if len(recent_data) >= 12:
                    ema_12 = recent_data['Close'].ewm(span=12).mean().iloc[-1]
                    ema_26 = recent_data['Close'].ewm(span=min(26, len(recent_data))).mean().iloc[-1]
                    macd = (ema_12 - ema_26) / ema_26
                else:
                    macd = 0
            except:
                macd = 0
            
            # Enhanced price trend (multiple timeframes)
            try:
                if len(recent_data) >= 15:
                    ma_5 = recent_data['Close'].rolling(5).mean().iloc[-1]
                    ma_15 = recent_data['Close'].rolling(15).mean().iloc[-1]
                    price_trend = (ma_5 - ma_15) / ma_15
                else:
                    price_trend = 0
            except:
                price_trend = 0
            
            # Enhanced volume analysis
            try:
                current_volume = data['Volume'].iloc[current_idx]
                avg_volume = recent_data['Volume'].rolling(15).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            except:
                volume_ratio = 1
            
            # Price momentum
            try:
                if len(recent_data) >= 10:
                    momentum = (current_price - recent_data['Close'].iloc[-10]) / recent_data['Close'].iloc[-10]
                else:
                    momentum = 0
            except:
                momentum = 0
            
            # Volatility measure
            try:
                if len(price_changes) >= 10:
                    volatility = price_changes.rolling(10).std().iloc[-1] / current_price
                else:
                    volatility = 0
            except:
                volatility = 0
            
            return {
                'rsi': rsi,
                'macd': macd,
                'price_trend': price_trend,
                'volume_ratio': volume_ratio,
                'momentum': momentum,
                'volatility': volatility
            }
            
        except Exception as e:
            return {
                'rsi': 50, 'macd': 0, 'price_trend': 0, 'volume_ratio': 1,
                'momentum': 0, 'volatility': 0
            }
    
    def train_model(self, stock_data, symbol, episodes=150):
        """Train the enhanced RL model"""
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
        """Simulate trading with enhanced logic"""
        cash = INITIAL_CAPITAL
        shares = 0
        total_reward = 0
        max_position_value = INITIAL_CAPITAL * 0.95  # Allow 95% allocation
        
        start_idx = 30
        end_idx = len(data) - 1
        
        for i in range(start_idx, end_idx):
            try:
                state_features = self.extract_features(data, i)
                
                if training:
                    action = self.agent.choose_action(state_features)
                else:
                    old_epsilon = self.agent.epsilon
                    self.agent.epsilon = 0
                    action = self.agent.choose_action(state_features)
                    self.agent.epsilon = old_epsilon
                
                current_price = data['Close'].iloc[i]
                next_price = data['Close'].iloc[i + 1]
                
                prev_value = cash + shares * current_price
                
                # Enhanced trading logic
                if action == 'BUY' and cash > current_price * 5:
                    current_position_value = shares * current_price
                    available_for_stocks = max_position_value - current_position_value
                    
                    if available_for_stocks > 0:
                        # More aggressive position sizing based on signals
                        rsi = state_features.get('rsi', 50)
                        trend = state_features.get('price_trend', 0)
                        momentum = state_features.get('momentum', 0)
                        
                        # Determine position size based on signal strength
                        signal_strength = 0
                        if rsi < 30: signal_strength += 0.3  # Oversold
                        if trend > 0.02: signal_strength += 0.3  # Strong uptrend
                        if momentum > 0.01: signal_strength += 0.2  # Positive momentum
                        
                        position_size = min(0.4 + signal_strength, 0.8)  # 40-80% of available cash
                        max_investment = min(cash * position_size, available_for_stocks)
                        
                        shares_to_buy = int(max_investment / current_price)
                        
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price
                            transaction_cost = cost * 0.0005  # Reduced transaction cost
                            total_cost = cost + transaction_cost
                            
                            if cash >= total_cost:
                                cash -= total_cost
                                shares += shares_to_buy
                        
                elif action == 'SELL' and shares > 0:
                    # Enhanced sell logic
                    rsi = state_features.get('rsi', 50)
                    trend = state_features.get('price_trend', 0)
                    momentum = state_features.get('momentum', 0)
                    
                    # Determine sell percentage based on signals
                    sell_signal = 0
                    if rsi > 70: sell_signal += 0.4  # Overbought
                    if trend < -0.02: sell_signal += 0.4  # Strong downtrend
                    if momentum < -0.01: sell_signal += 0.3  # Negative momentum
                    
                    sell_percentage = min(0.3 + sell_signal, 1.0)  # 30-100% of position
                    shares_to_sell = max(1, int(shares * sell_percentage))
                    
                    revenue = shares_to_sell * current_price
                    transaction_cost = revenue * 0.0005  # Reduced transaction cost
                    net_revenue = revenue - transaction_cost
                    
                    cash += net_revenue
                    shares -= shares_to_sell
                
                # Enhanced reward calculation
                new_value = cash + shares * next_price
                
                if prev_value > 0:
                    portfolio_return = (new_value - prev_value) / prev_value
                    market_return = (next_price - current_price) / current_price
                    
                    # More sophisticated reward function
                    reward = 0
                    
                    if action == 'BUY':
                        # Reward successful buy timing
                        if next_price > current_price * 1.002:  # Price went up
                            reward = portfolio_return * 200  # High reward for good timing
                        else:
                            reward = portfolio_return * 50   # Lower reward/penalty
                    
                    elif action == 'SELL':
                        # Reward successful sell timing
                        if next_price < current_price * 0.998:  # Price went down
                            reward = -portfolio_return * 200  # Reward for avoiding loss
                        else:
                            reward = -portfolio_return * 50   # Penalty for early sell
                    
                    else:  # HOLD
                        # Reward holding during sideways markets
                        if abs(market_return) < 0.01:  # Low volatility
                            reward = 1
                        else:
                            reward = portfolio_return * 20
                    
                    # Bonus for outperforming market
                    if portfolio_return > market_return:
                        reward += 2
                    
                    # Penalty for major losses
                    if portfolio_return < -0.02:
                        reward -= 5
                        
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
        """Predict trading action with enhanced confidence"""
        try:
            if len(stock_data) < 30:
                return "HOLD", 0.5
            
            if not self.trained:
                if self.agent.load_model(self.model_path):
                    self.trained = True
                else:
                    return "HOLD", 0.5
            
            features = self.extract_features(stock_data, len(stock_data) - 1)
            
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            action = self.agent.choose_action(features)
            self.agent.epsilon = old_epsilon
            
            # Enhanced confidence calculation
            state_key = self.agent.get_state_key(features)
            if state_key in self.agent.q_table:
                q_values = self.agent.q_table[state_key]
                max_q = max(q_values)
                min_q = min(q_values)
                
                if max_q != min_q:
                    confidence = (max_q - np.mean(q_values)) / (max_q - min_q)
                    confidence = 0.4 + (confidence * 0.5)  # Scale to 0.4-0.9
                else:
                    confidence = 0.6
                
                # Adjust confidence based on signal strength
                rsi = features.get('rsi', 50)
                trend = features.get('price_trend', 0)
                
                if (action == 'BUY' and rsi < 35 and trend > 0) or \
                   (action == 'SELL' and rsi > 65 and trend < 0):
                    confidence = min(confidence + 0.2, 0.95)
                
            else:
                confidence = 0.5
            
            return action, min(max(confidence, 0), 1)
            
        except Exception as e:
            return "HOLD", 0.5
    
    def backtest_strategy(self, stock_data, symbol):
        """Enhanced backtesting with realistic performance tracking"""
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
            
            start_idx = 30
            max_position_value = INITIAL_CAPITAL * 0.95
            
            for i in range(start_idx, len(stock_data)):
                try:
                    features = self.extract_features(stock_data, i)
                    
                    old_epsilon = self.agent.epsilon
                    self.agent.epsilon = 0
                    action = self.agent.choose_action(features)
                    self.agent.epsilon = old_epsilon
                    
                    current_price = stock_data['Close'].iloc[i]
                    
                    # Execute enhanced trading logic
                    if action == 'BUY' and cash > current_price * 5:
                        current_position_value = shares * current_price
                        available_for_stocks = max_position_value - current_position_value
                        
                        if available_for_stocks > 0:
                            # Signal-based position sizing
                            rsi = features.get('rsi', 50)
                            trend = features.get('price_trend', 0)
                            momentum = features.get('momentum', 0)
                            
                            signal_strength = 0
                            if rsi < 30: signal_strength += 0.3
                            if trend > 0.02: signal_strength += 0.3
                            if momentum > 0.01: signal_strength += 0.2
                            
                            position_size = min(0.4 + signal_strength, 0.8)
                            max_investment = min(cash * position_size, available_for_stocks)
                            shares_to_buy = int(max_investment / current_price)
                            
                            if shares_to_buy > 0:
                                cost = shares_to_buy * current_price
                                transaction_cost = cost * 0.0005
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
                        # Enhanced sell logic
                        rsi = features.get('rsi', 50)
                        trend = features.get('price_trend', 0)
                        momentum = features.get('momentum', 0)
                        
                        sell_signal = 0
                        if rsi > 70: sell_signal += 0.4
                        if trend < -0.02: sell_signal += 0.4
                        if momentum < -0.01: sell_signal += 0.3
                        
                        sell_percentage = min(0.3 + sell_signal, 1.0)
                        shares_to_sell = max(1, int(shares * sell_percentage))
                        
                        revenue = shares_to_sell * current_price
                        transaction_cost = revenue * 0.0005
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
                    portfolio_value = cash + shares * stock_data['Close'].iloc[i]
                    portfolio_values.append(portfolio_value)
                    actions_taken.append('HOLD')
            
            # Calculate enhanced performance metrics
            if len(portfolio_values) > 1:
                returns = pd.Series(portfolio_values).pct_change().dropna()
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                
                if returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                
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
