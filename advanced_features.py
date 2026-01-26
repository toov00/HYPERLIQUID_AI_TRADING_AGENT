#!/usr/bin/env python3
"""
Advanced features for Hyperliquid Optimizer
Include this module for additional analytics and signals
"""

import statistics
from typing import List, Dict
from datetime import datetime, timedelta


class AdvancedAnalytics:
    """Advanced trading analytics and signals"""
    
    @staticmethod
    def calculate_vwap(trades: List[Dict]) -> float:
        """Calculate Volume Weighted Average Price"""
        if not trades:
            return 0.0
        
        total_volume = sum(float(t['sz']) * float(t['px']) for t in trades)
        total_size = sum(float(t['sz']) for t in trades)
        
        return total_volume / total_size if total_size > 0 else 0.0
    
    @staticmethod
    def detect_order_imbalance(bids: List[Dict], asks: List[Dict], depth: int = 10) -> Dict:
        """Detect order book imbalance"""
        bid_volume = sum(float(level['sz']) for level in bids[:depth])
        ask_volume = sum(float(level['sz']) for level in asks[:depth])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return {'imbalance': 0, 'direction': 'neutral'}
        
        imbalance_ratio = (bid_volume - ask_volume) / total_volume
        
        direction = 'bullish' if imbalance_ratio > 0.2 else 'bearish' if imbalance_ratio < -0.2 else 'neutral'
        
        return {
            'imbalance': abs(imbalance_ratio),
            'direction': direction,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume
        }
    
    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 14) -> float:
        """Calculate price momentum"""
        if len(prices) < period:
            return 0.0
        
        recent_prices = prices[-period:]
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_changes = changes[-period:]
        
        gains = [c for c in recent_changes if c > 0]
        losses = [-c for c in recent_changes if c < 0]
        
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def detect_support_resistance(prices: List[float], tolerance: float = 0.001) -> Dict:
        """Identify potential support and resistance levels"""
        if len(prices) < 20:
            return {'support': [], 'resistance': []}
        
        # Find local minima (support) and maxima (resistance)
        supports = []
        resistances = []
        
        for i in range(2, len(prices) - 2):
            # Local minimum
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    supports.append(prices[i])
            
            # Local maximum
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    resistances.append(prices[i])
        
        # Cluster similar levels
        def cluster_levels(levels, tolerance):
            if not levels:
                return []
            
            sorted_levels = sorted(levels)
            clusters = [[sorted_levels[0]]]
            
            for level in sorted_levels[1:]:
                if abs(level - clusters[-1][-1]) / level <= tolerance:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            return [sum(c) / len(c) for c in clusters]
        
        return {
            'support': cluster_levels(supports, tolerance),
            'resistance': cluster_levels(resistances, tolerance)
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio for returns"""
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / std_return
    
    @staticmethod
    def detect_volume_spike(current_volume: float, avg_volume: float, threshold: float = 2.0) -> bool:
        """Detect if current volume is significantly higher than average"""
        if avg_volume == 0:
            return False
        
        return current_volume / avg_volume >= threshold
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0}
        
        recent_prices = prices[-period:]
        middle = statistics.mean(recent_prices)
        std = statistics.stdev(recent_prices)
        
        return {
            'upper': middle + (std_dev * std),
            'middle': middle,
            'lower': middle - (std_dev * std)
        }
    
    @staticmethod
    def time_to_trade(hour: int) -> Dict:
        """Suggest optimal trading times based on historical patterns"""
        # These are general patterns - adjust based on your market
        high_volume_hours = [13, 14, 15, 16, 17, 20, 21]  # UTC hours with typically high volume
        low_volume_hours = [0, 1, 2, 3, 4, 5, 6]  # UTC hours with typically low volume
        
        if hour in high_volume_hours:
            return {'score': 'high', 'message': 'High activity period - expect tighter spreads'}
        elif hour in low_volume_hours:
            return {'score': 'low', 'message': 'Low activity period - be cautious of wider spreads'}
        else:
            return {'score': 'medium', 'message': 'Moderate activity period'}


class TradingSignals:
    """Generate trading signals based on market conditions"""
    
    def __init__(self):
        self.analytics = AdvancedAnalytics()
    
    def generate_entry_signal(
        self,
        current_price: float,
        order_book: Dict,
        recent_trades: List[Dict],
        price_history: List[float]
    ) -> Dict:
        """Generate comprehensive entry signal"""
        signal = {
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'signals': [],
            'overall_score': 0
        }
        
        # Check spread
        bids = order_book.get('levels', [[], []])[0]
        asks = order_book.get('levels', [[], []])[1]
        
        if bids and asks:
            spread = float(asks[0]['px']) - float(bids[0]['px'])
            spread_pct = (spread / float(asks[0]['px'])) * 100
            
            if spread_pct < 0.1:
                signal['signals'].append({'type': 'tight_spread', 'score': 2})
                signal['overall_score'] += 2
        
        # Check order book imbalance
        if bids and asks:
            imbalance = self.analytics.detect_order_imbalance(bids, asks)
            if imbalance['imbalance'] > 0.3:
                signal['signals'].append({
                    'type': 'order_imbalance',
                    'direction': imbalance['direction'],
                    'score': 1
                })
                signal['overall_score'] += 1
        
        # Check RSI
        if len(price_history) >= 14:
            rsi = self.analytics.calculate_rsi(price_history)
            if rsi < 30:
                signal['signals'].append({'type': 'oversold', 'rsi': rsi, 'score': 2})
                signal['overall_score'] += 2
            elif rsi > 70:
                signal['signals'].append({'type': 'overbought', 'rsi': rsi, 'score': -2})
                signal['overall_score'] -= 2
        
        # Check volume
        if recent_trades:
            recent_volume = sum(float(t['sz']) * float(t['px']) for t in recent_trades[-10:])
            if recent_volume > 50000:
                signal['signals'].append({'type': 'high_volume', 'score': 1})
                signal['overall_score'] += 1
        
        # Overall assessment
        if signal['overall_score'] >= 3:
            signal['recommendation'] = 'STRONG BUY'
        elif signal['overall_score'] >= 1:
            signal['recommendation'] = 'BUY'
        elif signal['overall_score'] <= -3:
            signal['recommendation'] = 'STRONG SELL'
        elif signal['overall_score'] <= -1:
            signal['recommendation'] = 'SELL'
        else:
            signal['recommendation'] = 'NEUTRAL'
        
        return signal


# Example usage
if __name__ == "__main__":
    # Test the analytics
    analytics = AdvancedAnalytics()
    
    # Sample data
    sample_trades = [
        {'sz': '100', 'px': '25.50'},
        {'sz': '200', 'px': '25.55'},
        {'sz': '150', 'px': '25.52'}
    ]
    
    sample_prices = [25.0 + i * 0.1 for i in range(30)]
    
    print("VWAP:", analytics.calculate_vwap(sample_trades))
    print("RSI:", analytics.calculate_rsi(sample_prices))
    print("Bollinger Bands:", analytics.calculate_bollinger_bands(sample_prices))
    print("Support/Resistance:", analytics.detect_support_resistance(sample_prices))
    