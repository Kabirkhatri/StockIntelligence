"""
Technical analysis indicators and calculations
"""
import pandas as pd
import numpy as np
# import talib  # Commented out due to installation issues
import streamlit as st
from config.settings import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    MA_SHORT, MA_LONG, BOLLINGER_PERIOD, BOLLINGER_STD
)

class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_rsi(self, data, period=RSI_PERIOD):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': macd_signal,
            'Histogram': macd_histogram
        })
    
    def calculate_moving_averages(self, data, short_window=MA_SHORT, long_window=MA_LONG):
        """Calculate Simple and Exponential Moving Averages"""
        ma_data = pd.DataFrame(index=data.index)
        
        ma_data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
        ma_data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()
        ma_data[f'EMA_{short_window}'] = data['Close'].ewm(span=short_window).mean()
        ma_data[f'EMA_{long_window}'] = data['Close'].ewm(span=long_window).mean()
        
        return ma_data
    
    def calculate_bollinger_bands(self, data, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD):
        """Calculate Bollinger Bands"""
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        return pd.DataFrame({
            'BB_Upper': sma + (std * std_dev),
            'BB_Middle': sma,
            'BB_Lower': sma - (std * std_dev)
        })
    
    def calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        k_percent = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        })
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def calculate_volume_indicators(self, data):
        """Calculate volume-based indicators"""
        volume_indicators = pd.DataFrame(index=data.index)
        
        # Volume Moving Average
        volume_indicators['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        obv = []
        obv_value = 0
        for i in range(len(data)):
            if i == 0:
                obv.append(data['Volume'].iloc[i])
            else:
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv_value += data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv_value -= data['Volume'].iloc[i]
                obv.append(obv_value)
        
        volume_indicators['OBV'] = obv
        
        # Volume Rate of Change
        volume_indicators['Volume_ROC'] = data['Volume'].pct_change(periods=10) * 100
        
        return volume_indicators
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on technical indicators"""
        signals = pd.DataFrame(index=data.index)
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        macd = self.calculate_macd(data)
        ma = self.calculate_moving_averages(data)
        bb = self.calculate_bollinger_bands(data)
        stoch = self.calculate_stochastic(data)
        
        # RSI signals
        signals['RSI_Oversold'] = rsi < 30
        signals['RSI_Overbought'] = rsi > 70
        
        # MACD signals
        signals['MACD_Bullish'] = (macd['MACD'] > macd['Signal']) & (macd['MACD'].shift() <= macd['Signal'].shift())
        signals['MACD_Bearish'] = (macd['MACD'] < macd['Signal']) & (macd['MACD'].shift() >= macd['Signal'].shift())
        
        # Moving Average signals
        signals['MA_Bullish'] = ma[f'SMA_{MA_SHORT}'] > ma[f'SMA_{MA_LONG}']
        signals['MA_Bearish'] = ma[f'SMA_{MA_SHORT}'] < ma[f'SMA_{MA_LONG}']
        
        # Bollinger Bands signals
        signals['BB_Oversold'] = data['Close'] < bb['BB_Lower']
        signals['BB_Overbought'] = data['Close'] > bb['BB_Upper']
        
        # Stochastic signals
        signals['Stoch_Oversold'] = (stoch['Stoch_K'] < 20) & (stoch['Stoch_D'] < 20)
        signals['Stoch_Overbought'] = (stoch['Stoch_K'] > 80) & (stoch['Stoch_D'] > 80)
        
        # Composite signals
        buy_signals = (
            signals['RSI_Oversold'] | 
            signals['MACD_Bullish'] | 
            signals['BB_Oversold'] | 
            signals['Stoch_Oversold']
        )
        
        sell_signals = (
            signals['RSI_Overbought'] | 
            signals['MACD_Bearish'] | 
            signals['BB_Overbought'] | 
            signals['Stoch_Overbought']
        )
        
        signals['Buy_Signal'] = buy_signals
        signals['Sell_Signal'] = sell_signals
        signals['Signal_Score'] = buy_signals.astype(int) - sell_signals.astype(int)
        
        return signals
    
    def get_technical_summary(self, data):
        """Get a comprehensive technical analysis summary"""
        current_price = data['Close'].iloc[-1]
        
        # Calculate all indicators
        rsi = self.calculate_rsi(data).iloc[-1]
        macd = self.calculate_macd(data)
        ma = self.calculate_moving_averages(data)
        bb = self.calculate_bollinger_bands(data)
        stoch = self.calculate_stochastic(data)
        atr_series = self.calculate_atr(data)
        atr = atr_series.iloc[-1] if not atr_series.empty else 0
        
        # Get latest values
        macd_current = macd['MACD'].iloc[-1] if not macd.empty else 0
        macd_signal_val = macd['Signal'].iloc[-1] if not macd.empty else 0
        sma_20 = ma[f'SMA_{MA_SHORT}'].iloc[-1] if not ma.empty else current_price
        sma_50 = ma[f'SMA_{MA_LONG}'].iloc[-1] if not ma.empty else current_price
        bb_upper = bb['BB_Upper'].iloc[-1] if not bb.empty else current_price
        bb_lower = bb['BB_Lower'].iloc[-1] if not bb.empty else current_price
        stoch_k = stoch['Stoch_K'].iloc[-1] if not stoch.empty else 50
        
        # Generate interpretation
        summary = {
            'current_price': current_price,
            'rsi': rsi,
            'rsi_interpretation': self._interpret_rsi(rsi),
            'macd': macd_current,
            'macd_signal': macd_signal_val,
            'macd_interpretation': self._interpret_macd(macd_current, macd_signal_val),
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ma_interpretation': self._interpret_ma(current_price, sma_20, sma_50),
            'bb_position': self._interpret_bb_position(current_price, bb_upper, bb_lower),
            'stochastic': stoch_k,
            'stoch_interpretation': self._interpret_stochastic(stoch_k),
            'atr': atr,
            'volatility_level': self._interpret_volatility(atr, current_price)
        }
        
        return summary
    
    def _interpret_rsi(self, rsi):
        if rsi > 70:
            return "Overbought - Potential sell signal"
        elif rsi < 30:
            return "Oversold - Potential buy signal"
        else:
            return "Neutral"
    
    def _interpret_macd(self, macd, signal):
        if macd > signal:
            return "Bullish - MACD above signal line"
        else:
            return "Bearish - MACD below signal line"
    
    def _interpret_ma(self, price, sma_20, sma_50):
        if price > sma_20 > sma_50:
            return "Strong Bullish - Price above both MAs"
        elif price > sma_20 and sma_20 < sma_50:
            return "Weak Bullish - Price above short MA only"
        elif price < sma_20 < sma_50:
            return "Strong Bearish - Price below both MAs"
        else:
            return "Weak Bearish - Price below short MA only"
    
    def _interpret_bb_position(self, price, upper, lower):
        bb_width = upper - lower
        bb_position = (price - lower) / bb_width
        
        if bb_position > 0.8:
            return "Near upper band - Potentially overbought"
        elif bb_position < 0.2:
            return "Near lower band - Potentially oversold"
        else:
            return "Within normal range"
    
    def _interpret_stochastic(self, stoch_k):
        if stoch_k > 80:
            return "Overbought territory"
        elif stoch_k < 20:
            return "Oversold territory"
        else:
            return "Neutral range"
    
    def _interpret_volatility(self, atr, price):
        atr_percentage = (atr / price) * 100
        if atr_percentage > 3:
            return "High volatility"
        elif atr_percentage < 1:
            return "Low volatility"
        else:
            return "Moderate volatility"

# Create global instance
technical_analyzer = TechnicalAnalyzer()
