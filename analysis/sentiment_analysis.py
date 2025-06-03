"""
Sentiment analysis for Indian stock markets using news and social media data
"""
import pandas as pd
import numpy as np
import streamlit as st
import requests
from textblob import TextBlob
import re
from datetime import datetime, timedelta
from config.settings import NEWS_SOURCES
import time

class SentimentAnalyzer:
    def __init__(self):
        self.news_sources = NEWS_SOURCES
        
    def analyze_stock_sentiment(self, symbol, days_back=7):
        """
        Analyze sentiment for a specific stock
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back for news
        
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Fetch news articles
            news_data = self._fetch_news_articles(symbol, days_back)
            
            if not news_data:
                return self._get_neutral_sentiment()
            
            # Analyze sentiment
            sentiment_results = self._analyze_text_sentiment(news_data)
            
            # Calculate metrics
            sentiment_summary = self._calculate_sentiment_metrics(sentiment_results)
            
            return sentiment_summary
            
        except Exception as e:
            st.warning(f"Error in sentiment analysis for {symbol}: {str(e)}")
            return self._get_neutral_sentiment()
    
    def _fetch_news_articles(self, symbol, days_back):
        """
        Fetch news articles related to the stock
        This is a simplified implementation - in production, you'd use proper news APIs
        """
        # Simulated news headlines for demonstration
        # In production, you would use:
        # - News API (newsapi.org)
        # - Google News API
        # - Financial news RSS feeds
        # - Twitter API for social sentiment
        
        sample_headlines = [
            f"{symbol} reports strong quarterly results, beats expectations",
            f"Analysts upgrade {symbol} target price on robust fundamentals",
            f"{symbol} announces new product launch, market responds positively",
            f"Industry experts bullish on {symbol} long-term prospects",
            f"{symbol} stock shows resilience amid market volatility",
            f"Institutional investors increase stake in {symbol}",
            f"{symbol} management optimistic about future growth",
            f"Technical analysts see strong support levels for {symbol}",
            f"{symbol} dividend announcement boosts investor confidence",
            f"Market sentiment remains positive for {symbol} stock"
        ]
        
        # For demonstration, return sample headlines
        # In production, implement actual news fetching
        return sample_headlines
    
    def _analyze_text_sentiment(self, text_data):
        """
        Analyze sentiment of text data using TextBlob
        
        Args:
            text_data: List of text strings
        
        Returns:
            list: Sentiment scores for each text
        """
        sentiment_results = []
        
        for text in text_data:
            if text and isinstance(text, str):
                # Clean text
                cleaned_text = self._clean_text(text)
                
                # Analyze sentiment using TextBlob
                blob = TextBlob(cleaned_text)
                
                sentiment_score = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Categorize sentiment
                if sentiment_score > 0.1:
                    sentiment_label = "Positive"
                elif sentiment_score < -0.1:
                    sentiment_label = "Negative"
                else:
                    sentiment_label = "Neutral"
                
                sentiment_results.append({
                    'text': text,
                    'sentiment_score': sentiment_score,
                    'subjectivity': subjectivity,
                    'sentiment_label': sentiment_label
                })
        
        return sentiment_results
    
    def _clean_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _calculate_sentiment_metrics(self, sentiment_results):
        """
        Calculate comprehensive sentiment metrics
        
        Args:
            sentiment_results: List of sentiment analysis results
        
        Returns:
            dict: Sentiment metrics
        """
        if not sentiment_results:
            return self._get_neutral_sentiment()
        
        scores = [result['sentiment_score'] for result in sentiment_results]
        labels = [result['sentiment_label'] for result in sentiment_results]
        
        # Calculate basic metrics
        avg_sentiment = np.mean(scores)
        sentiment_std = np.std(scores)
        
        # Count sentiment categories
        positive_count = labels.count('Positive')
        negative_count = labels.count('Negative')
        neutral_count = labels.count('Neutral')
        total_count = len(labels)
        
        # Calculate percentages
        positive_pct = (positive_count / total_count) * 100
        negative_pct = (negative_count / total_count) * 100
        neutral_pct = (neutral_count / total_count) * 100
        
        # Overall sentiment classification
        if avg_sentiment > 0.2:
            overall_sentiment = "Bullish"
        elif avg_sentiment < -0.2:
            overall_sentiment = "Bearish"
        else:
            overall_sentiment = "Neutral"
        
        # Sentiment strength
        sentiment_strength = abs(avg_sentiment)
        if sentiment_strength > 0.5:
            strength_label = "Strong"
        elif sentiment_strength > 0.2:
            strength_label = "Moderate"
        else:
            strength_label = "Weak"
        
        # Sentiment consistency (lower std means more consistent)
        if sentiment_std < 0.2:
            consistency = "High"
        elif sentiment_std < 0.5:
            consistency = "Moderate"
        else:
            consistency = "Low"
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_sentiment,
            'sentiment_strength': strength_label,
            'consistency': consistency,
            'positive_percentage': positive_pct,
            'negative_percentage': negative_pct,
            'neutral_percentage': neutral_pct,
            'total_articles': total_count,
            'sentiment_distribution': {
                'Positive': positive_count,
                'Negative': negative_count,
                'Neutral': neutral_count
            },
            'raw_scores': scores
        }
    
    def _get_neutral_sentiment(self):
        """Return neutral sentiment when no data is available"""
        return {
            'overall_sentiment': 'Neutral',
            'sentiment_score': 0.0,
            'sentiment_strength': 'Weak',
            'consistency': 'Low',
            'positive_percentage': 33.3,
            'negative_percentage': 33.3,
            'neutral_percentage': 33.3,
            'total_articles': 0,
            'sentiment_distribution': {
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0
            },
            'raw_scores': []
        }
    
    def get_market_sentiment(self, symbols_list):
        """
        Get overall market sentiment for a list of stocks
        
        Args:
            symbols_list: List of stock symbols
        
        Returns:
            dict: Market sentiment summary
        """
        market_sentiments = []
        
        for symbol in symbols_list:
            sentiment = self.analyze_stock_sentiment(symbol)
            market_sentiments.append(sentiment['sentiment_score'])
        
        if market_sentiments:
            market_avg = np.mean(market_sentiments)
            
            if market_avg > 0.1:
                market_mood = "Bullish"
            elif market_avg < -0.1:
                market_mood = "Bearish"
            else:
                market_mood = "Neutral"
            
            return {
                'market_sentiment': market_mood,
                'market_score': market_avg,
                'analyzed_stocks': len(symbols_list),
                'sentiment_range': {
                    'min': min(market_sentiments),
                    'max': max(market_sentiments)
                }
            }
        
        return {
            'market_sentiment': 'Neutral',
            'market_score': 0.0,
            'analyzed_stocks': 0,
            'sentiment_range': {'min': 0, 'max': 0}
        }
    
    def analyze_earnings_sentiment(self, symbol, earnings_date):
        """
        Analyze sentiment around earnings announcements
        This would typically look at news sentiment before and after earnings
        """
        # Simplified implementation
        # In production, you would analyze sentiment trends around specific dates
        
        pre_earnings_sentiment = self.analyze_stock_sentiment(symbol, days_back=10)
        
        # Simulate post-earnings sentiment (in practice, fetch actual post-earnings news)
        post_earnings_modifier = np.random.normal(0, 0.2)  # Random sentiment change
        post_earnings_score = pre_earnings_sentiment['sentiment_score'] + post_earnings_modifier
        post_earnings_score = max(-1, min(1, post_earnings_score))  # Clamp between -1 and 1
        
        return {
            'pre_earnings': pre_earnings_sentiment,
            'post_earnings_score': post_earnings_score,
            'sentiment_change': post_earnings_score - pre_earnings_sentiment['sentiment_score'],
            'earnings_impact': 'Positive' if post_earnings_score > pre_earnings_sentiment['sentiment_score'] else 'Negative'
        }
    
    def get_sector_sentiment(self, sector_name, stock_symbols):
        """
        Analyze sentiment for an entire sector
        
        Args:
            sector_name: Name of the sector
            stock_symbols: List of stocks in the sector
        
        Returns:
            dict: Sector sentiment analysis
        """
        sector_sentiments = []
        
        for symbol in stock_symbols:
            sentiment = self.analyze_stock_sentiment(symbol)
            sector_sentiments.append({
                'symbol': symbol,
                'sentiment_score': sentiment['sentiment_score'],
                'overall_sentiment': sentiment['overall_sentiment']
            })
        
        # Calculate sector-wide metrics
        scores = [s['sentiment_score'] for s in sector_sentiments]
        sector_avg = np.mean(scores) if scores else 0
        
        if sector_avg > 0.1:
            sector_mood = "Bullish"
        elif sector_avg < -0.1:
            sector_mood = "Bearish"
        else:
            sector_mood = "Neutral"
        
        return {
            'sector_name': sector_name,
            'sector_sentiment': sector_mood,
            'sector_score': sector_avg,
            'stock_sentiments': sector_sentiments,
            'analyzed_stocks': len(stock_symbols)
        }

# Create global instance
sentiment_analyzer = SentimentAnalyzer()
