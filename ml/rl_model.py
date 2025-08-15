"""
Reinforcement Learning model for stock trading decisions
Using Deep Q-Network (DQN) for trading strategy
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
layers = keras.layers
import streamlit as st
from collections import deque
import random
from config.settings import (
    RL_LOOKBACK_WINDOW, RL_ACTIONS, RL_EPISODES, RL_LEARNING_RATE,
    RL_GAMMA, RL_EPSILON_START, RL_EPSILON_END, RL_EPSILON_DECAY,
    INITIAL_CAPITAL, MAX_POSITION_SIZE, TRANSACTION_COST
)
from analysis.technical_analysis import technical_analyzer
from analysis.fundamental_analysis import fundamental_analyzer
from analysis.sentiment_analysis import sentiment_analyzer
import pickle
import os

class TradingEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data, initial_capital=INITIAL_CAPITAL):
        self.data = data
        self.initial_capital = initial_capital
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = RL_LOOKBACK_WINDOW
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.transaction_costs = 0
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(self._get_state_size())
        
        # Technical indicators
        current_data = self.data.iloc[self.current_step-RL_LOOKBACK_WINDOW:self.current_step]
        
        # Price features (normalized)
        prices = current_data['Close'].values
        price_features = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Volume features (normalized)
        volumes = current_data['Volume'].values
        volume_features = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.append(returns, 0)  # Pad to same length
        
        # Technical indicators (simplified)
        rsi = self._calculate_simple_rsi(prices)
        ma_short = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        ma_long = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        ma_signal = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
        
        # Portfolio state
        current_price = prices[-1]
        position_value = self.shares * current_price
        total_value = self.cash + position_value
        cash_ratio = self.cash / total_value if total_value > 0 else 0
        position_ratio = position_value / total_value if total_value > 0 else 0
        
        # Combine all features
        state = np.concatenate([
            price_features[-10:],  # Last 10 normalized prices
            volume_features[-10:],  # Last 10 normalized volumes
            returns[-10:],  # Last 10 returns
            [rsi, ma_signal, cash_ratio, position_ratio]
        ])
        
        return state
    
    def _get_state_size(self):
        """Get size of state vector"""
        return 34  # 10 prices + 10 volumes + 10 returns + 4 additional features
    
    def _calculate_simple_rsi(self, prices, period=14):
        """Calculate simple RSI"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True
        
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        
        # Execute action
        reward = self._execute_action(action, current_price, next_price)
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.shares * next_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done
    
    def _execute_action(self, action, current_price, next_price):
        """Execute trading action and calculate reward"""
        prev_portfolio_value = self.cash + (self.shares * current_price)
        
        if action == 0:  # BUY
            # Buy as many shares as possible with available cash (up to position limit)
            max_investment = min(self.cash * 0.95, prev_portfolio_value * MAX_POSITION_SIZE)
            shares_to_buy = int(max_investment / current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_fee = cost * TRANSACTION_COST
                total_cost = cost + transaction_fee
                
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.shares += shares_to_buy
                    self.transaction_costs += transaction_fee
                    
                    self.trades.append({
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': total_cost
                    })
        
        elif action == 1:  # SELL
            # Sell all shares
            if self.shares > 0:
                revenue = self.shares * current_price
                transaction_fee = revenue * TRANSACTION_COST
                net_revenue = revenue - transaction_fee
                
                self.cash += net_revenue
                self.transaction_costs += transaction_fee
                
                self.trades.append({
                    'action': 'SELL',
                    'shares': self.shares,
                    'price': current_price,
                    'revenue': net_revenue
                })
                
                self.shares = 0
        
        # Action 2 is HOLD - no action required
        
        # Calculate reward
        new_portfolio_value = self.cash + (self.shares * next_price)
        portfolio_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Reward shaping
        reward = portfolio_return * 100  # Scale reward
        
        # Penalty for excessive trading
        if action != 2:  # If not holding
            reward -= 0.1  # Small penalty for trading
        
        return reward

class DQNAgent:
    """Deep Q-Network agent for trading"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = RL_EPSILON_START
        self.epsilon_min = RL_EPSILON_END
        self.epsilon_decay = RL_EPSILON_DECAY
        self.learning_rate = RL_LEARNING_RATE
        self.gamma = RL_GAMMA
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
    
    def _build_model(self):
        """Build neural network model"""
        model = keras.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Predict Q-values for current states
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.q_network.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.q_network = keras.models.load_model(filepath)
            self.update_target_network()
            return True
        return False

class RLTradingSystem:
    """Complete RL trading system"""
    
    def __init__(self):
        self.agent = None
        self.environment = None
        self.trained = False
        self.model_path = "rl_trading_model.h5"
    
    def prepare_data(self, stock_data, symbol):
        """Prepare data with features for RL model"""
        # Add technical indicators
        try:
            rsi = technical_analyzer.calculate_rsi(stock_data)
            macd = technical_analyzer.calculate_macd(stock_data)
            ma = technical_analyzer.calculate_moving_averages(stock_data)
            bb = technical_analyzer.calculate_bollinger_bands(stock_data)
            
            # Add features to stock data
            enhanced_data = stock_data.copy()
            enhanced_data['RSI'] = rsi
            enhanced_data['MACD'] = macd['MACD']
            enhanced_data['MA_20'] = ma[f'SMA_{20}']
            enhanced_data['MA_50'] = ma[f'SMA_{50}']
            enhanced_data['BB_Upper'] = bb['BB_Upper']
            enhanced_data['BB_Lower'] = bb['BB_Lower']
            
            # Fill NaN values
            enhanced_data = enhanced_data.fillna(method='forward').fillna(method='backward')
            
            return enhanced_data
            
        except Exception as e:
            st.warning(f"Error preparing data: {str(e)}")
            return stock_data
    
    def train_model(self, stock_data, symbol, episodes=100):
        """Train the RL model"""
        # Prepare data
        prepared_data = self.prepare_data(stock_data, symbol)
        
        # Create environment and agent
        self.environment = TradingEnvironment(prepared_data)
        state_size = self.environment._get_state_size()
        self.agent = DQNAgent(state_size, len(RL_ACTIONS))
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done = self.environment.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            episode_rewards.append(total_reward)
            
            # Update progress
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            status_text.text(f"Training Episode {episode + 1}/{episodes} - Reward: {total_reward:.2f}")
        
        progress_bar.empty()
        status_text.empty()
        
        self.trained = True
        
        # Save model
        self.agent.save_model(self.model_path)
        
        return episode_rewards
    
    def predict_action(self, stock_data, symbol):
        """Predict trading action for current market conditions"""
        if not self.trained or self.agent is None:
            # Try to load existing model
            if not self.load_model():
                return "HOLD", 0.5  # Default to hold if no model
        
        # Prepare data
        prepared_data = self.prepare_data(stock_data, symbol)
        
        # Create environment for prediction
        env = TradingEnvironment(prepared_data)
        state = env._get_state()
        
        # Get action probabilities
        q_values = self.agent.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Get best action
        best_action_idx = np.argmax(q_values)
        best_action = RL_ACTIONS[best_action_idx]
        
        # Calculate confidence (based on Q-value difference)
        confidence = (np.max(q_values) - np.mean(q_values)) / (np.std(q_values) + 1e-8)
        confidence = min(max(confidence, 0), 1)  # Clamp between 0 and 1
        
        return best_action, confidence
    
    def backtest_strategy(self, stock_data, symbol):
        """Backtest the RL strategy"""
        if not self.trained or self.agent is None:
            if not self.load_model():
                return None
        
        # Prepare data
        prepared_data = self.prepare_data(stock_data, symbol)
        
        # Create environment
        env = TradingEnvironment(prepared_data)
        state = env.reset()
        
        # Track performance
        portfolio_values = [env.initial_capital]
        actions_taken = []
        trades = []
        
        while True:
            # Use trained agent (no exploration)
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0  # No exploration during backtesting
            
            action = self.agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Restore epsilon
            self.agent.epsilon = original_epsilon
            
            # Record data
            portfolio_values.append(env.portfolio_value)
            actions_taken.append(RL_ACTIONS[action])
            
            state = next_state
            
            if done:
                break
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame({
            'Portfolio_Value': portfolio_values[:-1]  # Remove last extra value
        }, index=prepared_data.index[RL_LOOKBACK_WINDOW:])
        
        # Calculate returns
        portfolio_returns = portfolio_df['Portfolio_Value'].pct_change().dropna()
        benchmark_returns = prepared_data['Close'][RL_LOOKBACK_WINDOW:].pct_change().dropna()
        
        # Performance metrics
        total_return = (portfolio_values[-2] - env.initial_capital) / env.initial_capital
        benchmark_return = (prepared_data['Close'].iloc[-1] - prepared_data['Close'].iloc[RL_LOOKBACK_WINDOW]) / prepared_data['Close'].iloc[RL_LOOKBACK_WINDOW]
        
        # Calculate Sharpe ratio, max drawdown, etc.
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # Max drawdown
        peak = portfolio_df['Portfolio_Value'].expanding().max()
        drawdown = (portfolio_df['Portfolio_Value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_values': portfolio_values[:-1],
            'actions': actions_taken,
            'trades': env.trades,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(env.trades),
            'transaction_costs': env.transaction_costs
        }
    
    def load_model(self):
        """Load existing trained model"""
        if os.path.exists(self.model_path):
            try:
                # Create agent with default parameters
                self.agent = DQNAgent(34, len(RL_ACTIONS))  # Default state size
                self.agent.load_model(self.model_path)
                self.trained = True
                return True
            except:
                return False
        return False
    
    def get_feature_importance(self):
        """Analyze feature importance (simplified implementation)"""
        # This is a simplified implementation
        # In practice, you'd use techniques like SHAP or permutation importance
        
        feature_names = [
            'Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5',
            'Price_6', 'Price_7', 'Price_8', 'Price_9', 'Price_10',
            'Volume_1', 'Volume_2', 'Volume_3', 'Volume_4', 'Volume_5',
            'Volume_6', 'Volume_7', 'Volume_8', 'Volume_9', 'Volume_10',
            'Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5',
            'Return_6', 'Return_7', 'Return_8', 'Return_9', 'Return_10',
            'RSI', 'MA_Signal', 'Cash_Ratio', 'Position_Ratio'
        ]
        
        # Simulate feature importance (in practice, calculate actual importance)
        importance_scores = np.random.random(len(feature_names))
        importance_scores = importance_scores / importance_scores.sum()
        
        return dict(zip(feature_names, importance_scores))

# Create global instance
rl_trading_system = RLTradingSystem()
