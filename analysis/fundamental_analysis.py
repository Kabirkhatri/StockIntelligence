"""
Fundamental analysis for Indian stocks
"""
import pandas as pd
import numpy as np
import streamlit as st
from data.stock_data import stock_fetcher

class FundamentalAnalyzer:
    def __init__(self):
        self.benchmark_metrics = {
            'pe_ratio': {'excellent': 15, 'good': 25, 'average': 35},
            'price_to_book': {'excellent': 1.5, 'good': 3, 'average': 5},
            'debt_to_equity': {'excellent': 0.3, 'good': 0.6, 'average': 1.0},
            'roe': {'excellent': 20, 'good': 15, 'average': 10},
            'revenue_growth': {'excellent': 20, 'good': 10, 'average': 5},
            'profit_margin': {'excellent': 15, 'good': 10, 'average': 5}
        }
    
    def analyze_fundamentals(self, symbol, exchange="NSE"):
        """
        Perform comprehensive fundamental analysis
        
        Args:
            symbol: Stock symbol
            exchange: Exchange (NSE/BSE)
        
        Returns:
            dict: Fundamental analysis results
        """
        try:
            # Fetch fundamental data
            stock_info = stock_fetcher.fetch_stock_info(symbol, exchange)
            
            if not stock_info:
                return None
            
            # Calculate derived metrics
            analysis = self._calculate_fundamental_metrics(stock_info)
            
            # Generate scores and recommendations
            analysis['scores'] = self._calculate_scores(stock_info)
            analysis['overall_score'] = self._calculate_overall_score(analysis['scores'])
            analysis['recommendation'] = self._generate_recommendation(analysis['overall_score'])
            analysis['risk_level'] = self._assess_risk_level(stock_info)
            analysis['growth_potential'] = self._assess_growth_potential(stock_info)
            
            return analysis
            
        except Exception as e:
            st.error(f"Error in fundamental analysis for {symbol}: {str(e)}")
            return None
    
    def _calculate_fundamental_metrics(self, stock_info):
        """Calculate additional fundamental metrics"""
        metrics = {}
        
        # Valuation metrics
        metrics['market_cap_category'] = self._categorize_market_cap(stock_info.get('market_cap', 0))
        
        # Efficiency ratios
        if stock_info.get('roe', 0) and stock_info.get('profit_margin', 0):
            # Asset turnover approximation
            metrics['asset_turnover'] = stock_info['roe'] / (stock_info['profit_margin'] * 100)
        
        # Growth metrics
        if stock_info.get('revenue_growth'):
            metrics['revenue_growth_category'] = self._categorize_growth(stock_info['revenue_growth'] * 100)
        
        # Financial health
        metrics['financial_strength'] = self._assess_financial_strength(stock_info)
        
        # Dividend analysis
        if stock_info.get('dividend_yield'):
            metrics['dividend_category'] = self._categorize_dividend_yield(stock_info['dividend_yield'] * 100)
        
        return metrics
    
    def _calculate_scores(self, stock_info):
        """Calculate scores for different fundamental aspects"""
        scores = {}
        
        # Valuation Score (0-100)
        pe_score = self._score_metric(stock_info.get('pe_ratio', 0), 'pe_ratio', reverse=True)
        pb_score = self._score_metric(stock_info.get('price_to_book', 0), 'price_to_book', reverse=True)
        scores['valuation'] = (pe_score + pb_score) / 2
        
        # Profitability Score (0-100)
        roe_score = self._score_metric(stock_info.get('roe', 0) * 100 if stock_info.get('roe') else 0, 'roe')
        margin_score = self._score_metric(stock_info.get('profit_margin', 0) * 100 if stock_info.get('profit_margin') else 0, 'profit_margin')
        scores['profitability'] = (roe_score + margin_score) / 2
        
        # Growth Score (0-100)
        growth_score = self._score_metric(stock_info.get('revenue_growth', 0) * 100 if stock_info.get('revenue_growth') else 0, 'revenue_growth')
        scores['growth'] = growth_score
        
        # Financial Health Score (0-100)
        debt_score = self._score_metric(stock_info.get('debt_to_equity', 0), 'debt_to_equity', reverse=True)
        scores['financial_health'] = debt_score
        
        return scores
    
    def _score_metric(self, value, metric_type, reverse=False):
        """Score a metric based on benchmark values"""
        if value == 0 or np.isnan(value):
            return 0
        
        benchmarks = self.benchmark_metrics.get(metric_type, {})
        if not benchmarks:
            return 50  # Neutral score if no benchmarks
        
        excellent = benchmarks['excellent']
        good = benchmarks['good']
        average = benchmarks['average']
        
        if reverse:
            # Lower values are better (e.g., P/E ratio, debt-to-equity)
            if value <= excellent:
                return 100
            elif value <= good:
                return 80
            elif value <= average:
                return 60
            else:
                return max(0, 40 - (value - average) * 2)
        else:
            # Higher values are better (e.g., ROE, profit margin)
            if value >= excellent:
                return 100
            elif value >= good:
                return 80
            elif value >= average:
                return 60
            else:
                return max(0, value * 2)
    
    def _calculate_overall_score(self, scores):
        """Calculate weighted overall score"""
        weights = {
            'valuation': 0.25,
            'profitability': 0.35,
            'growth': 0.25,
            'financial_health': 0.15
        }
        
        weighted_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
        return weighted_score
    
    def _generate_recommendation(self, overall_score):
        """Generate investment recommendation based on overall score"""
        if overall_score >= 80:
            return "Strong Buy"
        elif overall_score >= 70:
            return "Buy"
        elif overall_score >= 60:
            return "Hold"
        elif overall_score >= 40:
            return "Weak Hold"
        else:
            return "Sell"
    
    def _categorize_market_cap(self, market_cap):
        """Categorize company by market cap"""
        if market_cap >= 20000000000000:  # 2 lakh crore
            return "Large Cap"
        elif market_cap >= 5000000000000:  # 50,000 crore
            return "Mid Cap"
        else:
            return "Small Cap"
    
    def _categorize_growth(self, growth_rate):
        """Categorize growth rate"""
        if growth_rate >= 20:
            return "High Growth"
        elif growth_rate >= 10:
            return "Moderate Growth"
        elif growth_rate >= 0:
            return "Low Growth"
        else:
            return "Declining"
    
    def _categorize_dividend_yield(self, dividend_yield):
        """Categorize dividend yield"""
        if dividend_yield >= 4:
            return "High Dividend"
        elif dividend_yield >= 2:
            return "Moderate Dividend"
        elif dividend_yield > 0:
            return "Low Dividend"
        else:
            return "No Dividend"
    
    def _assess_financial_strength(self, stock_info):
        """Assess overall financial strength"""
        debt_to_equity = stock_info.get('debt_to_equity', 0)
        roe = stock_info.get('roe', 0)
        profit_margin = stock_info.get('profit_margin', 0)
        
        if debt_to_equity < 0.3 and roe and roe > 0.15 and profit_margin and profit_margin > 0.1:
            return "Strong"
        elif debt_to_equity < 0.6 and roe and roe > 0.1:
            return "Good"
        elif debt_to_equity < 1.0:
            return "Average"
        else:
            return "Weak"
    
    def _assess_risk_level(self, stock_info):
        """Assess investment risk level"""
        beta = stock_info.get('beta', 1.0)
        debt_to_equity = stock_info.get('debt_to_equity', 0)
        
        risk_factors = 0
        
        if beta > 1.5:
            risk_factors += 2
        elif beta > 1.2:
            risk_factors += 1
        
        if debt_to_equity > 1.0:
            risk_factors += 2
        elif debt_to_equity > 0.6:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "High Risk"
        elif risk_factors >= 1:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _assess_growth_potential(self, stock_info):
        """Assess growth potential"""
        revenue_growth = stock_info.get('revenue_growth', 0)
        forward_pe = stock_info.get('forward_pe', 0)
        trailing_pe = stock_info.get('pe_ratio', 0)
        
        growth_score = 0
        
        if revenue_growth and revenue_growth > 0.15:
            growth_score += 2
        elif revenue_growth and revenue_growth > 0.05:
            growth_score += 1
        
        if forward_pe and trailing_pe and forward_pe < trailing_pe:
            growth_score += 1
        
        if growth_score >= 2:
            return "High Growth Potential"
        elif growth_score >= 1:
            return "Moderate Growth Potential"
        else:
            return "Limited Growth Potential"
    
    def compare_stocks(self, symbols, exchange="NSE"):
        """
        Compare fundamental metrics of multiple stocks
        
        Args:
            symbols: List of stock symbols
            exchange: Exchange name
        
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for symbol in symbols:
            stock_info = stock_fetcher.fetch_stock_info(symbol, exchange)
            if stock_info:
                comparison_data.append({
                    'Symbol': symbol,
                    'Company': stock_info.get('company_name', symbol),
                    'Market Cap (Cr)': stock_info.get('market_cap', 0) / 10000000 if stock_info.get('market_cap') else 0,
                    'P/E Ratio': stock_info.get('pe_ratio', 0),
                    'P/B Ratio': stock_info.get('price_to_book', 0),
                    'ROE (%)': stock_info.get('roe', 0) * 100 if stock_info.get('roe') else 0,
                    'Debt/Equity': stock_info.get('debt_to_equity', 0),
                    'Revenue Growth (%)': stock_info.get('revenue_growth', 0) * 100 if stock_info.get('revenue_growth') else 0,
                    'Dividend Yield (%)': stock_info.get('dividend_yield', 0) * 100 if stock_info.get('dividend_yield') else 0
                })
        
        return pd.DataFrame(comparison_data)
    
    def get_peer_comparison(self, symbol, sector, exchange="NSE"):
        """
        Get peer comparison for a stock (simplified implementation)
        In production, you would fetch sector-specific stocks from a database
        """
        # This is a simplified implementation
        # In practice, you'd have a database of stocks by sector
        popular_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
        
        return self.compare_stocks(popular_stocks[:5], exchange)

# Create global instance
fundamental_analyzer = FundamentalAnalyzer()
