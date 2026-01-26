#!/usr/bin/env python3
"""
Hyperliquid AI Trading Agent
Uses Claude AI to make intelligent trading decisions and send personalized alerts
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import websockets
from discord_webhook import DiscordWebhook, DiscordEmbed
import requests
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperliquidAIAgent:
    """AI-powered trading agent using Claude for intelligent market analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.coin = config.get('coin', 'HYPE')
        self.pair = f"{self.coin}/USDC"
        
        # Initialize Claude AI
        self.anthropic = Anthropic(api_key=config.get('anthropic_api_key'))
        self.model = config.get('claude_model', 'claude-sonnet-4-20250514')
        
        # AI agent settings
        self.ai_analysis_interval = config.get('ai_analysis_interval', 60)  # Analyze every 60 seconds
        self.user_preferences = config.get('user_preferences', {
            'trading_style': 'balanced',  # conservative, balanced, aggressive
            'risk_tolerance': 'medium',  # low, medium, high
            'position_size': 'medium',  # small, medium, large
            'time_horizon': 'intraday'  # scalping, intraday, swing
        })
        
        # Learning from user feedback
        self.user_feedback = []
        self.alert_history = []
        
        # Data storage
        self.current_book = None
        self.recent_trades = []
        self.price_history = []
        self.last_ai_analysis = None
        self.last_analysis_time = 0
        
        # Alert cooldown (seconds)
        self.alert_cooldown = config.get('alert_cooldown', 300)  # 5 minutes
        
    async def connect(self):
        """Connect to Hyperliquid WebSocket and start monitoring"""
        logger.info(f"Connecting to Hyperliquid WebSocket for {self.pair}...")
        
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
            "subscription": {
                "type": "l2Book",
                "coin": self.coin
            }
        }
        await websocket.send(json.dumps(subscription))
        logger.info(f"Subscribed to L2 order book for {self.coin}")
    
    async def _subscribe_trades(self, websocket):
        """Subscribe to trades data"""
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": self.coin
            }
        }
        await websocket.send(json.dumps(subscription))
        logger.info(f"Subscribed to trades for {self.coin}")
    
    async def _subscribe_allmids(self, websocket):
        """Subscribe to all mid prices"""
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "allMids"
            }
        }
        await websocket.send(json.dumps(subscription))
        logger.info("Subscribed to all mid prices")
    
    async def _process_messages(self, websocket):
        """Process incoming WebSocket messages"""
        async for message in websocket:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
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
    
    async def _perform_ai_analysis(self):
        """Use Claude AI to analyze market conditions and make recommendations"""
        try:
            # Gather current market data
            market_data = self._gather_market_data()
            
            # Create prompt for Claude
            prompt = self._create_analysis_prompt(market_data)
            
            logger.info("🤖 Requesting AI analysis from Claude...")
            
            # Call Claude API
            message = self.anthropic.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract analysis
            analysis = message.content[0].text
            self.last_ai_analysis = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'market_data': market_data
            }
            
            logger.info("✅ AI analysis completed")
            
            # Parse and act on analysis
            await self._process_ai_analysis(analysis, market_data)
            
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
                'avg_size': sum(float(t['sz']) for t in recent_20) / len(recent_20),
                'buy_sell_ratio': self._calculate_buy_sell_ratio(recent_20)
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
        
        return data
    
    def _calculate_imbalance(self, bids: List[Dict], asks: List[Dict]) -> float:
        """Calculate order book imbalance"""
        bid_volume = sum(float(level['sz']) for level in bids[:10])
        ask_volume = sum(float(level['sz']) for level in asks[:10])
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0
    
    def _calculate_buy_sell_ratio(self, trades: List[Dict]) -> float:
        """Calculate buy/sell pressure from recent trades"""
        buys = sum(1 for t in trades if t.get('side') == 'buy')
        sells = len(trades) - buys
        return buys / sells if sells > 0 else 1.0
    
    def _create_analysis_prompt(self, market_data: Dict) -> str:
        """Create a detailed prompt for Claude to analyze market conditions"""
        return f"""You are an expert cryptocurrency trading analyst with deep knowledge of market microstructure, order flow, and optimal trade execution.

Analyze the current market conditions for {self.pair} on Hyperliquid DEX and provide actionable trading insights.

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

USER TRADING PROFILE:
- Trading Style: {self.user_preferences.get('trading_style', 'balanced')}
- Risk Tolerance: {self.user_preferences.get('risk_tolerance', 'medium')}
- Typical Position Size: {self.user_preferences.get('position_size', 'medium')}
- Time Horizon: {self.user_preferences.get('time_horizon', 'intraday')}

PREVIOUS ALERT HISTORY (last 5):
{json.dumps(self.alert_history[-5:], indent=2) if self.alert_history else 'No previous alerts'}

YOUR TASK:
Analyze the market conditions and provide:

1. **Market Assessment** (2-3 sentences): What's happening right now?

2. **Trading Opportunity Score** (0-10): How favorable are current conditions for entering/exiting positions?
   - Consider: spread, liquidity, volatility, order book imbalance, recent momentum
   - 0-3: Poor conditions, avoid trading
   - 4-6: Neutral conditions, proceed with caution
   - 7-10: Excellent conditions, favorable for execution

3. **Specific Recommendation**: Should the user:
   - STRONG BUY SIGNAL - Excellent entry conditions
   - BUY SIGNAL - Good entry conditions  
   - NEUTRAL - Wait for better conditions
   - SELL SIGNAL - Good exit conditions
   - STRONG SELL SIGNAL - Excellent exit conditions
   - NO SIGNAL - Conditions not notable

4. **Reasoning** (3-4 bullet points): Why this recommendation?

5. **Optimal Execution Strategy**: If trading, what's the best approach?
   - Market order vs limit order?
   - Estimated slippage for different sizes
   - Any timing considerations?

6. **Risk Factors** (if any): What should the user watch out for?

Respond in JSON format:
{{
  "market_assessment": "string",
  "opportunity_score": number,
  "recommendation": "string",
  "reasoning": ["point1", "point2", "point3"],
  "execution_strategy": "string",
  "risk_factors": "string",
  "alert_user": boolean,
  "alert_priority": "low|medium|high"
}}

Be concise, actionable, and honest about uncertainty."""
    
    async def _process_ai_analysis(self, analysis: str, market_data: Dict):
        """Parse AI analysis and take appropriate actions"""
        try:
            # Try to parse JSON from analysis
            # Claude might wrap it in markdown, so extract it
            if "```json" in analysis:
                json_str = analysis.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis:
                json_str = analysis.split("```")[1].split("```")[0].strip()
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
            
            # Decide whether to alert user
            should_alert = ai_recommendation.get('alert_user', False)
            opportunity_score = ai_recommendation.get('opportunity_score', 0)
            
            if should_alert and opportunity_score >= 7:
                await self._send_ai_alert(ai_recommendation, market_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI analysis as JSON: {e}")
            logger.info(f"Raw analysis: {analysis}")
    
    async def _send_ai_alert(self, recommendation: Dict, market_data: Dict):
        """Send intelligent alert based on AI analysis"""
        rec_type = recommendation.get('recommendation', 'UNKNOWN')
        priority = recommendation.get('alert_priority', 'medium')
        
        # Determine alert color and emoji
        alert_config = {
            'STRONG BUY SIGNAL': {'color': '00ff00', 'emoji': '🚀'},
            'BUY SIGNAL': {'color': '32cd32', 'emoji': '📈'},
            'STRONG SELL SIGNAL': {'color': 'ff4500', 'emoji': '⚠️'},
            'SELL SIGNAL': {'color': 'ffa500', 'emoji': '📉'},
            'NEUTRAL': {'color': 'ffcc00', 'emoji': '⏸️'}
        }
        
        config = alert_config.get(rec_type, {'color': '808080', 'emoji': 'ℹ️'})
        
        title = f"{config['emoji']} AI Trading Signal: {rec_type}"
        description = recommendation.get('market_assessment', 'No assessment available')
        
        # Send to Discord
        if self.config.get('discord_webhook'):
            await self._send_discord_ai_alert(
                title, 
                description, 
                recommendation, 
                market_data,
                config['color']
            )
        
        # Send to Telegram
        if self.config.get('telegram_bot_token') and self.config.get('telegram_chat_id'):
            await self._send_telegram_ai_alert(title, description, recommendation, market_data)
        
        logger.info(f"🔔 AI Alert sent: {rec_type}")
    
    async def _send_discord_ai_alert(
        self, 
        title: str, 
        description: str,
        recommendation: Dict,
        market_data: Dict,
        color: str
    ):
        """Send AI-powered alert to Discord"""
        webhook = DiscordWebhook(
            url=self.config['discord_webhook'],
            username="Hyperliquid AI Agent 🤖"
        )
        
        embed = DiscordEmbed(
            title=title,
            description=description,
            color=color
        )
        
        # Add key metrics
        embed.add_embed_field(
            name="📊 Opportunity Score", 
            value=f"{recommendation.get('opportunity_score', 0)}/10",
            inline=True
        )
        
        if 'order_book' in market_data:
            ob = market_data['order_book']
            embed.add_embed_field(
                name="💰 Current Price",
                value=f"${ob['best_bid']:.4f} - ${ob['best_ask']:.4f}",
                inline=True
            )
            embed.add_embed_field(
                name="📏 Spread",
                value=f"{ob['spread_pct']:.3f}%",
                inline=True
            )
        
        # Add reasoning
        reasoning = recommendation.get('reasoning', [])
        if reasoning:
            embed.add_embed_field(
                name="🧠 AI Reasoning",
                value="\n".join(f"• {r}" for r in reasoning[:3]),
                inline=False
            )
        
        # Add execution strategy
        if recommendation.get('execution_strategy'):
            embed.add_embed_field(
                name="⚡ Execution Strategy",
                value=recommendation['execution_strategy'],
                inline=False
            )
        
        # Add risk factors
        if recommendation.get('risk_factors'):
            embed.add_embed_field(
                name="⚠️ Risk Factors",
                value=recommendation['risk_factors'],
                inline=False
            )
        
        embed.set_footer(text=f"Powered by Claude AI • {self.pair}")
        embed.set_timestamp()
        
        webhook.add_embed(embed)
        
        try:
            webhook.execute()
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    async def _send_telegram_ai_alert(
        self, 
        title: str, 
        description: str,
        recommendation: Dict,
        market_data: Dict
    ):
        """Send AI-powered alert to Telegram"""
        message = f"*{title}*\n\n{description}\n\n"
        message += f"*Opportunity Score:* {recommendation.get('opportunity_score', 0)}/10\n\n"
        
        reasoning = recommendation.get('reasoning', [])
        if reasoning:
            message += "*AI Reasoning:*\n"
            for r in reasoning[:3]:
                message += f"• {r}\n"
            message += "\n"
        
        if recommendation.get('execution_strategy'):
            message += f"*Strategy:* {recommendation['execution_strategy']}\n\n"
        
        message += f"_Powered by Claude AI • {datetime.now().strftime('%H:%M:%S UTC')}_"
        
        url = f"https://api.telegram.org/bot{self.config['telegram_bot_token']}/sendMessage"
        payload = {
            'chat_id': self.config['telegram_chat_id'],
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    async def _handle_trades(self, data: List[Dict]):
        """Process trade updates"""
        for trade in data:
            if trade.get('coin') != self.coin:
                continue
            
            self.recent_trades.append(trade)
            
            # Keep only last 100 trades
            if len(self.recent_trades) > 100:
                self.recent_trades.pop(0)
            
            price = float(trade['px'])
            size = float(trade['sz'])
            side = trade.get('side', 'unknown')
            
            logger.debug(f"Trade: {side.upper()} {size:.4f} {self.coin} @ ${price:.4f}")
    
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


async def main():
    """Main entry point"""
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found. Please create one from config.example.json")
        return
    
    # Validate API key
    if not config.get('anthropic_api_key'):
        logger.error("anthropic_api_key not found in config.json")
        logger.error("Please add your Claude API key to use the AI agent")
        return
    
    # Initialize and run AI agent
    agent = HyperliquidAIAgent(config)
    
    logger.info("🤖 Hyperliquid AI Agent starting...")
    logger.info(f"   Model: {agent.model}")
    logger.info(f"   Pair: {agent.pair}")
    logger.info(f"   AI Analysis Interval: {agent.ai_analysis_interval}s")
    
    # Run with auto-reconnect
    while True:
        try:
            await agent.connect()
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
    