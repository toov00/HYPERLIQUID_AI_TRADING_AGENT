#!/usr/bin/env python3
"""
Hyperliquid AI Trading Agent 
Powered by Groq (FREE!)
Uses Groq's free API with Llama 3.1 for intelligent market analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import websockets
from discord_webhook import DiscordWebhook, DiscordEmbed
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperliquidGroqAgent:
    """AI-powered trading agent using Groq's FREE API"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.coin = config.get('coin', 'HYPE')
        self.pair = f"{self.coin}/USDC"
        
        # Groq API configuration
        self.groq_api_key = config.get('groq_api_key')
        if not self.groq_api_key:
            raise ValueError("groq_api_key not found in config.json")
        
        self.groq_model = config.get('groq_model', 'llama-3.1-8b-instant')
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # AI agent settings
        self.ai_analysis_interval = config.get('ai_analysis_interval', 60)
        self.user_preferences = config.get('user_preferences', {
            'trading_style': 'balanced',
            'risk_tolerance': 'medium',
            'position_size': 'medium',
            'time_horizon': 'intraday'
        })
        
        # Data storage
        self.current_book = None
        self.recent_trades = []
        self.price_history = []
        self.last_ai_analysis = None
        self.last_analysis_time = 0
        self.alert_history = []
        
        # Alert cooldown
        self.alert_cooldown = config.get('alert_cooldown', 300)
    
    async def connect(self):
        """Connect to Hyperliquid WebSocket and start monitoring"""
        logger.info(f"Starting Groq-Powered AI Agent for {self.pair}")
        logger.info(f"   Model: {self.groq_model}")
        logger.info(f"   Analysis Interval: {self.ai_analysis_interval}s")
        logger.info(f"   Status: FREE API - No limits!")
        
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to order book (L2)
            await self._subscribe_orderbook(websocket)
            
            # Subscribe to trades
            await self._subscribe_trades(websocket)
            
            # Subscribe to all mids for price tracking
            await self._subscribe_allmids(websocket)
            
            logger.info(f"Successfully subscribed to {self.pair} data feeds")
            
            # Main message processing loop
            await self._process_messages(websocket)
    
    async def _subscribe_orderbook(self, websocket):
        """Subscribe to L2 order book data"""
        subscription = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": self.coin}
        }
        await websocket.send(json.dumps(subscription))
    
    async def _subscribe_trades(self, websocket):
        """Subscribe to trades data"""
        subscription = {
            "method": "subscribe",
            "subscription": {"type": "trades", "coin": self.coin}
        }
        await websocket.send(json.dumps(subscription))
    
    async def _subscribe_allmids(self, websocket):
        """Subscribe to all mid prices"""
        subscription = {
            "method": "subscribe",
            "subscription": {"type": "allMids"}
        }
        await websocket.send(json.dumps(subscription))
    
    async def _process_messages(self, websocket):
        """Process incoming WebSocket messages"""
        async for message in websocket:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, data: Dict):
        """Route messages to appropriate handlers"""
        if 'channel' not in data:
            return
        
        channel = data['channel']
        message_data = data.get('data', {})
        
        if channel == 'l2Book':
            await self._handle_orderbook(message_data)
        elif channel == 'trades':
            await self._handle_trades(message_data)
        elif channel == 'allMids':
            await self._handle_mids(message_data)
    
    async def _handle_orderbook(self, data: Dict):
        """Process order book updates with AI analysis"""
        if data.get('coin') != self.coin:
            return
        
        self.current_book = data
        levels = data.get('levels', [[], []])
        bids, asks = levels[0], levels[1]
        
        if not bids or not asks:
            return
        
        # Calculate basic metrics
        best_bid = float(bids[0]['px'])
        best_ask = float(asks[0]['px'])
        spread = best_ask - best_bid
        spread_pct = (spread / best_ask) * 100
        
        bid_depth = sum(float(level['sz']) for level in bids[:10])
        ask_depth = sum(float(level['sz']) for level in asks[:10])
        total_depth = bid_depth + ask_depth
        
        # Log current state
        logger.info(
            f"{self.pair} | Bid: ${best_bid:.4f} | Ask: ${best_ask:.4f} | "
            f"Spread: {spread_pct:.3f}% | Depth: ${total_depth:.2f}"
        )
        
        # Check if it's time for AI analysis
        current_time = datetime.now().timestamp()
        if current_time - self.last_analysis_time >= self.ai_analysis_interval:
            await self._perform_ai_analysis()
            self.last_analysis_time = current_time
    
    async def _handle_trades(self, data: List[Dict]):
        """Process trade updates"""
        for trade in data:
            if trade.get('coin') != self.coin:
                continue
            
            self.recent_trades.append(trade)
            
            # Keep only last 100 trades
            if len(self.recent_trades) > 100:
                self.recent_trades.pop(0)
    
    async def _handle_mids(self, data: Dict):
        """Process mid price updates"""
        mids = data.get('mids', {})
        
        if self.coin in mids:
            mid_price = float(mids[self.coin])
            self.price_history.append({
                'price': mid_price,
                'timestamp': datetime.now()
            })
            
            # Keep only last 100 prices
            if len(self.price_history) > 100:
                self.price_history.pop(0)
    
    async def _perform_ai_analysis(self):
        """Use Groq AI to analyze market conditions"""
        try:
            # Gather current market data
            market_data = self._gather_market_data()
            
            # Create prompt
            prompt = self._create_analysis_prompt(market_data)
            
            logger.info(f"Requesting AI analysis from Groq ({self.groq_model})...")
            
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.groq_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency trading analyst. Always respond with valid JSON only, no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(
                self.groq_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Groq API error ({response.status_code}): {error_detail}")
                try:
                    error_json = response.json()
                    logger.error(f"Error details: {json.dumps(error_json, indent=2)}")
                except:
                    logger.error(f"Error response: {error_detail}")
                logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
                response.raise_for_status()
            
            analysis = response.json()['choices'][0]['message']['content']
            
            self.last_ai_analysis = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'market_data': market_data
            }
            
            logger.info("AI analysis completed")
            
            # Parse and act on analysis
            await self._process_ai_analysis(analysis, market_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_json, indent=2)}")
                except:
                    logger.error(f"Error response: {e.response.text}")
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
    
    def _gather_market_data(self) -> Dict:
        """Gather current market conditions for AI analysis"""
        data = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat()
        }
        
        # Order book data
        if self.current_book:
            levels = self.current_book.get('levels', [[], []])
            bids, asks = levels[0], levels[1]
            
            if bids and asks:
                best_bid = float(bids[0]['px'])
                best_ask = float(asks[0]['px'])
                
                data['order_book'] = {
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': best_ask - best_bid,
                    'spread_pct': ((best_ask - best_bid) / best_ask) * 100,
                    'bid_depth_10': sum(float(level['sz']) for level in bids[:10]),
                    'ask_depth_10': sum(float(level['sz']) for level in asks[:10]),
                    'bid_ask_imbalance': self._calculate_imbalance(bids, asks)
                }
        
        # Recent trades data
        if self.recent_trades:
            recent_20 = self.recent_trades[-20:]
            data['trades'] = {
                'count': len(recent_20),
                'total_volume': sum(float(t['sz']) * float(t['px']) for t in recent_20),
                'avg_size': sum(float(t['sz']) for t in recent_20) / len(recent_20) if recent_20 else 0
            }
        
        # Price history data
        if len(self.price_history) >= 20:
            prices = [p['price'] for p in self.price_history[-60:]]
            data['price_action'] = {
                'current': prices[-1],
                'high_60': max(prices),
                'low_60': min(prices),
                'volatility': ((max(prices) - min(prices)) / prices[-1]) * 100,
                'trend': 'bullish' if prices[-1] > prices[0] else 'bearish'
            }
        
        # Pattern recognition
        data['patterns'] = self._detect_patterns()
        
        return data
    
    def _calculate_imbalance(self, bids: List[Dict], asks: List[Dict]) -> float:
        """Calculate order book imbalance"""
        bid_volume = sum(float(level['sz']) for level in bids[:10])
        ask_volume = sum(float(level['sz']) for level in asks[:10])
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0
    
    # ==================== PATTERN RECOGNITION ====================
    
    def _detect_patterns(self) -> Dict:
        """Detect market patterns from collected data"""
        patterns = {
            'detected': [],
            'trade_patterns': self._detect_trade_patterns(),
            'orderbook_patterns': self._detect_orderbook_patterns(),
            'price_patterns': self._detect_price_patterns(),
            'momentum': self._calculate_momentum()
        }
        
        # Summarize detected patterns (trade patterns first for priority)
        for category in ['trade_patterns', 'orderbook_patterns', 'price_patterns']:
            if patterns[category].get('signals'):
                patterns['detected'].extend(patterns[category]['signals'])
        
        return patterns
    
    def _detect_price_patterns(self) -> Dict:
        """Detect price-based patterns"""
        result = {'signals': []}
        
        if len(self.price_history) < 20:
            return result
        
        prices = [p['price'] for p in self.price_history]
        
        # Support and resistance levels
        support, resistance = self._find_support_resistance(prices)
        result['support'] = support
        result['resistance'] = resistance
        
        current = prices[-1]
        
        # Check proximity to support/resistance
        if support and abs(current - support) / current < 0.005:
            result['signals'].append('NEAR_SUPPORT')
        if resistance and abs(current - resistance) / current < 0.005:
            result['signals'].append('NEAR_RESISTANCE')
        
        # Breakout detection
        if len(prices) >= 30:
            recent_high = max(prices[-30:-5]) if len(prices) > 5 else max(prices[:-5])
            recent_low = min(prices[-30:-5]) if len(prices) > 5 else min(prices[:-5])
            
            if current > recent_high * 1.002:
                result['signals'].append('BREAKOUT_UP')
                result['breakout_pct'] = ((current - recent_high) / recent_high) * 100
            elif current < recent_low * 0.998:
                result['signals'].append('BREAKOUT_DOWN')
                result['breakout_pct'] = ((recent_low - current) / recent_low) * 100
        
        # Trend detection (higher highs/lows or lower highs/lows)
        if len(prices) >= 10:
            trend = self._detect_trend(prices[-10:])
            result['trend'] = trend
            if trend == 'strong_uptrend':
                result['signals'].append('STRONG_UPTREND')
            elif trend == 'strong_downtrend':
                result['signals'].append('STRONG_DOWNTREND')
        
        # Consolidation detection (low volatility)
        if len(prices) >= 20:
            recent_range = (max(prices[-20:]) - min(prices[-20:])) / prices[-1] * 100
            if recent_range < 0.5:
                result['signals'].append('CONSOLIDATION')
                result['consolidation_range'] = recent_range
        
        return result
    
    def _find_support_resistance(self, prices: List[float]) -> tuple:
        """Find support and resistance levels from price history"""
        if len(prices) < 10:
            return None, None
        
        # Simple approach: find local minima and maxima
        lows = []
        highs = []
        
        for i in range(2, len(prices) - 2):
            # Local minimum
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                lows.append(prices[i])
            # Local maximum
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                highs.append(prices[i])
        
        support = sum(lows[-3:]) / len(lows[-3:]) if lows else min(prices)
        resistance = sum(highs[-3:]) / len(highs[-3:]) if highs else max(prices)
        
        return round(support, 4), round(resistance, 4)
    
    def _detect_trend(self, prices: List[float]) -> str:
        """Detect trend from price sequence"""
        if len(prices) < 4:
            return 'neutral'
        
        # Compare first half vs second half
        mid = len(prices) // 2
        first_half_avg = sum(prices[:mid]) / mid
        second_half_avg = sum(prices[mid:]) / (len(prices) - mid)
        
        change_pct = (second_half_avg - first_half_avg) / first_half_avg * 100
        
        # Count higher highs and higher lows for uptrend
        higher_highs = 0
        higher_lows = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                higher_highs += 1
            else:
                higher_lows += 1
        
        trend_strength = higher_highs / len(prices)
        
        if change_pct > 1 and trend_strength > 0.6:
            return 'strong_uptrend'
        elif change_pct > 0.3:
            return 'uptrend'
        elif change_pct < -1 and trend_strength < 0.4:
            return 'strong_downtrend'
        elif change_pct < -0.3:
            return 'downtrend'
        return 'neutral'
    
    def _detect_orderbook_patterns(self) -> Dict:
        """Detect patterns in the order book"""
        result = {'signals': []}
        
        if not self.current_book:
            return result
        
        levels = self.current_book.get('levels', [[], []])
        bids, asks = levels[0], levels[1]
        
        if not bids or not asks:
            return result
        
        # Calculate average order size
        bid_sizes = [float(b['sz']) for b in bids[:20]]
        ask_sizes = [float(a['sz']) for a in asks[:20]]
        
        avg_bid_size = sum(bid_sizes) / len(bid_sizes) if bid_sizes else 0
        avg_ask_size = sum(ask_sizes) / len(ask_sizes) if ask_sizes else 0
        
        # Detect large walls (orders > 3x average)
        bid_walls = []
        ask_walls = []
        
        for i, bid in enumerate(bids[:20]):
            size = float(bid['sz'])
            if size > avg_bid_size * 3:
                bid_walls.append({
                    'price': float(bid['px']),
                    'size': size,
                    'multiple': size / avg_bid_size
                })
        
        for i, ask in enumerate(asks[:20]):
            size = float(ask['sz'])
            if size > avg_ask_size * 3:
                ask_walls.append({
                    'price': float(ask['px']),
                    'size': size,
                    'multiple': size / avg_ask_size
                })
        
        if bid_walls:
            result['bid_walls'] = bid_walls
            result['signals'].append('BID_WALL_DETECTED')
        
        if ask_walls:
            result['ask_walls'] = ask_walls
            result['signals'].append('ASK_WALL_DETECTED')
        
        # Detect thin book (low liquidity)
        total_bid_depth = sum(bid_sizes[:10])
        total_ask_depth = sum(ask_sizes[:10])
        
        if total_bid_depth < 100 or total_ask_depth < 100:
            result['signals'].append('THIN_BOOK')
            result['liquidity_warning'] = True
        
        # Detect heavy imbalance
        imbalance = self._calculate_imbalance(bids, asks)
        result['imbalance'] = imbalance
        
        if imbalance > 0.4:
            result['signals'].append('HEAVY_BID_PRESSURE')
        elif imbalance < -0.4:
            result['signals'].append('HEAVY_ASK_PRESSURE')
        
        # Detect spread widening (compare to recent)
        best_bid = float(bids[0]['px'])
        best_ask = float(asks[0]['px'])
        spread_pct = (best_ask - best_bid) / best_ask * 100
        
        if spread_pct > 0.1:
            result['signals'].append('WIDE_SPREAD')
            result['spread_pct'] = spread_pct
        elif spread_pct < 0.02:
            result['signals'].append('TIGHT_SPREAD')
            result['spread_pct'] = spread_pct
        
        return result
    
    def _detect_trade_patterns(self) -> Dict:
        """Detect patterns in recent trades"""
        result = {'signals': []}
        
        if len(self.recent_trades) < 10:
            return result
        
        recent = self.recent_trades[-50:]
        
        # Analyze trade sizes
        sizes = [float(t['sz']) for t in recent]
        avg_size = sum(sizes) / len(sizes)
        
        # Detect whale trades (> 5x average)
        whale_trades = []
        for t in recent[-20:]:
            size = float(t['sz'])
            if size > avg_size * 5:
                whale_trades.append({
                    'size': size,
                    'price': float(t['px']),
                    'side': t.get('side', 'unknown'),
                    'multiple': size / avg_size
                })
        
        if whale_trades:
            result['whale_trades'] = whale_trades
            result['signals'].append('WHALE_ACTIVITY')
        
        # Analyze buy/sell pressure from recent trades
        buy_volume = 0
        sell_volume = 0
        
        for t in recent[-20:]:
            size = float(t['sz']) * float(t['px'])
            side = t.get('side', '')
            if side == 'B' or side == 'buy':
                buy_volume += size
            elif side == 'A' or side == 'sell':
                sell_volume += size
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
            result['buy_ratio'] = buy_ratio
            
            if buy_ratio > 0.65:
                result['signals'].append('STRONG_BUYING_PRESSURE')
            elif buy_ratio < 0.35:
                result['signals'].append('STRONG_SELLING_PRESSURE')
        
        # Detect volume spike
        if len(self.recent_trades) >= 50:
            old_volume = sum(float(t['sz']) for t in self.recent_trades[-50:-25])
            new_volume = sum(float(t['sz']) for t in self.recent_trades[-25:])
            
            if old_volume > 0 and new_volume > old_volume * 2:
                result['signals'].append('VOLUME_SPIKE')
                result['volume_increase'] = (new_volume / old_volume - 1) * 100
        
        # Detect rapid price movement
        if len(recent) >= 10:
            prices = [float(t['px']) for t in recent[-10:]]
            price_change = (prices[-1] - prices[0]) / prices[0] * 100
            
            if abs(price_change) > 0.5:
                if price_change > 0:
                    result['signals'].append('RAPID_PRICE_INCREASE')
                else:
                    result['signals'].append('RAPID_PRICE_DECREASE')
                result['rapid_change_pct'] = price_change
        
        return result
    
    def _calculate_momentum(self) -> Dict:
        """Calculate price momentum indicators"""
        result = {}
        
        if len(self.price_history) < 20:
            return result
        
        prices = [p['price'] for p in self.price_history]
        
        # Short-term momentum (last 10 prices)
        if len(prices) >= 10:
            short_change = (prices[-1] - prices[-10]) / prices[-10] * 100
            result['short_term'] = round(short_change, 3)
        
        # Medium-term momentum (last 30 prices)
        if len(prices) >= 30:
            medium_change = (prices[-1] - prices[-30]) / prices[-30] * 100
            result['medium_term'] = round(medium_change, 3)
        
        # Rate of change
        if len(prices) >= 5:
            roc = (prices[-1] - prices[-5]) / prices[-5] * 100
            result['roc_5'] = round(roc, 3)
        
        # Simple moving average comparison
        if len(prices) >= 20:
            sma_10 = sum(prices[-10:]) / 10
            sma_20 = sum(prices[-20:]) / 20
            
            result['sma_10'] = round(sma_10, 4)
            result['sma_20'] = round(sma_20, 4)
            result['above_sma_10'] = prices[-1] > sma_10
            result['above_sma_20'] = prices[-1] > sma_20
            
            # Golden/death cross approximation
            if sma_10 > sma_20 * 1.005:
                result['sma_signal'] = 'bullish'
            elif sma_10 < sma_20 * 0.995:
                result['sma_signal'] = 'bearish'
            else:
                result['sma_signal'] = 'neutral'
        
        return result
    
    def _create_analysis_prompt(self, market_data: Dict) -> str:
        """Create analysis prompt for Groq"""
        return f"""Analyze the current market conditions for {self.pair} on Hyperliquid DEX and provide a trading recommendation.

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

USER TRADING PROFILE:
- Trading Style: {self.user_preferences.get('trading_style', 'balanced')}
- Risk Tolerance: {self.user_preferences.get('risk_tolerance', 'medium')}
- Position Size: {self.user_preferences.get('position_size', 'medium')}
- Time Horizon: {self.user_preferences.get('time_horizon', 'intraday')}

PREVIOUS ALERTS (last 3):
{json.dumps(self.alert_history[-3:], indent=2) if self.alert_history else 'No previous alerts'}

Provide your analysis in this EXACT JSON format (no markdown, no code blocks):

{{
  "market_assessment": "Brief 2-3 sentence summary of current conditions",
  "opportunity_score": 8,
  "recommendation": "STRONG BUY",
  "reasoning": [
    "First key point about why this is a good/bad opportunity",
    "Second key point about market conditions",
    "Third key point about execution quality"
  ],
  "execution_strategy": "Specific advice on how to execute (market vs limit, size, timing)",
  "risk_factors": "Key risks to be aware of",
  "alert_user": true,
  "alert_priority": "high"
}}

SCORING GUIDE:
- 0-3: Poor conditions, avoid trading
- 4-6: Neutral conditions, wait for better setup
- 7-8: Good opportunity, favorable conditions
- 9-10: Excellent opportunity, act quickly

RECOMMENDATIONS:
- STRONG BUY: Exceptional entry conditions (score 9-10)
- BUY: Good entry conditions (score 7-8)
- NEUTRAL: Wait for better conditions (score 4-6)
- SELL: Good exit conditions (score 7-8)
- STRONG SELL: Urgent exit conditions (score 9-10)

Consider: spread tightness, order book depth, liquidity, volatility, order imbalance, recent momentum, and user's trading style.

PATTERN SIGNALS TO WATCH FOR:
- NEAR_SUPPORT/NEAR_RESISTANCE: Price approaching key levels
- BREAKOUT_UP/BREAKOUT_DOWN: Price breaking recent range
- BID_WALL/ASK_WALL: Large orders that may act as support/resistance
- WHALE_ACTIVITY: Large trades detected
- STRONG_BUYING/SELLING_PRESSURE: Directional trade flow
- VOLUME_SPIKE: Unusual trading activity
- CONSOLIDATION: Low volatility, potential breakout setup

Weight these patterns heavily in your analysis. Multiple bullish signals together = higher opportunity score.

Respond ONLY with the JSON object, nothing else."""
    
    async def _process_ai_analysis(self, analysis: str, market_data: Dict):
        """Parse AI analysis and take appropriate actions"""
        try:
            # Clean up the response
            analysis = analysis.strip()
            
            # Remove markdown code blocks if present
            if "```json" in analysis:
                analysis = analysis.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis:
                analysis = analysis.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            start = analysis.find('{')
            end = analysis.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = analysis[start:end]
            else:
                json_str = analysis
            
            ai_recommendation = json.loads(json_str)
            
            # Save to history
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                'recommendation': ai_recommendation.get('recommendation'),
                'score': ai_recommendation.get('opportunity_score'),
                'alerted': ai_recommendation.get('alert_user', False)
            })
            
            # Keep only last 10 alerts in history
            if len(self.alert_history) > 10:
                self.alert_history.pop(0)
            
            # Decide whether to alert user
            should_alert = ai_recommendation.get('alert_user', False)
            opportunity_score = ai_recommendation.get('opportunity_score', 0)
            
            if should_alert and opportunity_score >= 7:
                await self._send_ai_alert(ai_recommendation, market_data)
            else:
                logger.info(f"   Score: {opportunity_score}/10 - {ai_recommendation.get('recommendation')} (not alerting)")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI analysis as JSON: {e}")
            logger.debug(f"Raw analysis: {analysis}")
        except Exception as e:
            logger.error(f"Error processing analysis: {e}")
    
    async def _send_ai_alert(self, recommendation: Dict, market_data: Dict):
        """Send intelligent alert based on AI analysis"""
        rec_type = recommendation.get('recommendation', 'UNKNOWN')
        
        # Determine alert configuration
        alert_config = {
            'STRONG BUY': {'color': '00ff00'},
            'BUY': {'color': '32cd32'},
            'STRONG SELL': {'color': 'ff4500'},
            'SELL': {'color': 'ffa500'},
            'NEUTRAL': {'color': 'ffcc00'}
        }
        
        config = alert_config.get(rec_type, {'color': '808080'})
        
        title = f"AI Signal: {rec_type}"
        description = recommendation.get('market_assessment', '')
        
        # Send to Discord
        if self.config.get('discord_webhook'):
            await self._send_discord_alert(title, description, recommendation, market_data, config['color'])
        
        logger.info(f"AI Alert sent: {rec_type} (Score: {recommendation.get('opportunity_score')}/10)")
    
    async def _send_discord_alert(self, title: str, description: str, recommendation: Dict, market_data: Dict, color: str):
        """Send AI alert to Discord"""
        webhook = DiscordWebhook(
            url=self.config['discord_webhook'],
            username="Hyperliquid AI Agent"
        )
        
        embed = DiscordEmbed(title=title, description=description, color=color)
        
        embed.add_embed_field(
            name="Opportunity Score",
            value=f"**{recommendation.get('opportunity_score', 0)}/10**",
            inline=True
        )
        
        if 'order_book' in market_data:
            ob = market_data['order_book']
            embed.add_embed_field(
                name="Price Range",
                value=f"${ob['best_bid']:.4f} - ${ob['best_ask']:.4f}",
                inline=True
            )
            embed.add_embed_field(
                name="Spread",
                value=f"{ob['spread_pct']:.3f}%",
                inline=True
            )
        
        reasoning = recommendation.get('reasoning', [])
        if reasoning:
            embed.add_embed_field(
                name="AI Reasoning",
                value="\n".join(f"• {r}" for r in reasoning[:3]),
                inline=False
            )
        
        if recommendation.get('execution_strategy'):
            embed.add_embed_field(
                name="Execution Strategy",
                value=recommendation['execution_strategy'],
                inline=False
            )
        
        if recommendation.get('risk_factors'):
            embed.add_embed_field(
                name="Risk Factors",
                value=recommendation['risk_factors'],
                inline=False
            )
        
        embed.set_footer(text=f"Powered by Groq ({self.groq_model}) • {self.pair} • FREE AI")
        embed.set_timestamp()
        
        webhook.add_embed(embed)
        
        try:
            webhook.execute()
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")


async def main():
    """Main entry point"""
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found. Please create one from config.example.json")
        return
    
    # Validate Groq API key
    if not config.get('groq_api_key'):
        logger.error("groq_api_key not found in config.json")
        logger.error("")
        logger.error("Get your FREE Groq API key:")
        logger.error("1. Go to https://console.groq.com/")
        logger.error("2. Sign up (no credit card required)")
        logger.error("3. Get your API key")
        logger.error("4. Add it to config.json as 'groq_api_key'")
        return
    
    # Initialize and run Groq AI agent
    agent = HyperliquidGroqAgent(config)
    
    logger.info("=" * 60)
    logger.info("GROQ-POWERED AI TRADING AGENT")
    logger.info("=" * 60)
    logger.info("100% FREE - No credit card, no limits!")
    logger.info(f"Model: {config.get('groq_model', 'llama-3.1-8b-instant')}")
    logger.info("Lightning-fast inference")
    logger.info("=" * 60)
    
    # Run with auto-reconnect
    while True:
        try:
            await agent.connect()
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("Shutting down agent...")
            break
        except Exception as e:
            logger.error(f"Error: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nAgent stopped by user")
        