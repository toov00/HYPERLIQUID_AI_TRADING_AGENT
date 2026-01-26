#!/usr/bin/env python3
"""
Hyperliquid AI Trading Agent - Powered by Groq (FREE!)
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
        
        self.groq_model = config.get('groq_model', 'llama-3.1-70b-versatile')
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
        logger.info(f"🚀 Starting Groq-Powered AI Agent for {self.pair}")
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
            
            logger.info(f"✅ Successfully subscribed to {self.pair} data feeds")
            
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
            
            logger.info("🤖 Requesting AI analysis from Groq (Llama 3.1)...")
            
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
            response.raise_for_status()
            
            analysis = response.json()['choices'][0]['message']['content']
            
            self.last_ai_analysis = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'market_data': market_data
            }
            
            logger.info("✅ AI analysis completed")
            
            # Parse and act on analysis
            await self._process_ai_analysis(analysis, market_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
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
        
        return data
    
    def _calculate_imbalance(self, bids: List[Dict], asks: List[Dict]) -> float:
        """Calculate order book imbalance"""
        bid_volume = sum(float(level['sz']) for level in bids[:10])
        ask_volume = sum(float(level['sz']) for level in asks[:10])
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0
    
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
            'STRONG BUY': {'color': '00ff00', 'emoji': '🚀'},
            'BUY': {'color': '32cd32', 'emoji': '📈'},
            'STRONG SELL': {'color': 'ff4500', 'emoji': '⚠️'},
            'SELL': {'color': 'ffa500', 'emoji': '📉'},
            'NEUTRAL': {'color': 'ffcc00', 'emoji': '⏸️'}
        }
        
        config = alert_config.get(rec_type, {'color': '808080', 'emoji': 'ℹ️'})
        
        title = f"{config['emoji']} AI Signal: {rec_type}"
        description = recommendation.get('market_assessment', '')
        
        # Send to Discord
        if self.config.get('discord_webhook'):
            await self._send_discord_alert(title, description, recommendation, market_data, config['color'])
        
        # Send to Telegram
        if self.config.get('telegram_bot_token') and self.config.get('telegram_chat_id'):
            await self._send_telegram_alert(title, description, recommendation, market_data)
        
        logger.info(f"🔔 AI Alert sent: {rec_type} (Score: {recommendation.get('opportunity_score')}/10)")
    
    async def _send_discord_alert(self, title: str, description: str, recommendation: Dict, market_data: Dict, color: str):
        """Send AI alert to Discord"""
        webhook = DiscordWebhook(
            url=self.config['discord_webhook'],
            username="Hyperliquid AI Agent 🤖"
        )
        
        embed = DiscordEmbed(title=title, description=description, color=color)
        
        embed.add_embed_field(
            name="📊 Opportunity Score",
            value=f"**{recommendation.get('opportunity_score', 0)}/10**",
            inline=True
        )
        
        if 'order_book' in market_data:
            ob = market_data['order_book']
            embed.add_embed_field(
                name="💰 Price Range",
                value=f"${ob['best_bid']:.4f} - ${ob['best_ask']:.4f}",
                inline=True
            )
            embed.add_embed_field(
                name="📏 Spread",
                value=f"{ob['spread_pct']:.3f}%",
                inline=True
            )
        
        reasoning = recommendation.get('reasoning', [])
        if reasoning:
            embed.add_embed_field(
                name="🧠 AI Reasoning",
                value="\n".join(f"• {r}" for r in reasoning[:3]),
                inline=False
            )
        
        if recommendation.get('execution_strategy'):
            embed.add_embed_field(
                name="⚡ Execution Strategy",
                value=recommendation['execution_strategy'],
                inline=False
            )
        
        if recommendation.get('risk_factors'):
            embed.add_embed_field(
                name="⚠️ Risk Factors",
                value=recommendation['risk_factors'],
                inline=False
            )
        
        embed.set_footer(text=f"Powered by Groq (Llama 3.1) • {self.pair} • FREE AI")
        embed.set_timestamp()
        
        webhook.add_embed(embed)
        
        try:
            webhook.execute()
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    async def _send_telegram_alert(self, title: str, description: str, recommendation: Dict, market_data: Dict):
        """Send AI alert to Telegram"""
        message = f"*{title}*\n\n{description}\n\n"
        message += f"*Opportunity Score:* {recommendation.get('opportunity_score', 0)}/10\n\n"
        
        reasoning = recommendation.get('reasoning', [])
        if reasoning:
            message += "*AI Reasoning:*\n"
            for r in reasoning[:3]:
                message += f"• {r}\n"
            message += "\n"
        
        if recommendation.get('execution_strategy'):
            message += f"*Execution Strategy:*\n{recommendation['execution_strategy']}\n\n"
        
        if recommendation.get('risk_factors'):
            message += f"*Risk Factors:*\n{recommendation['risk_factors']}\n\n"
        
        message += f"_Powered by Groq (Llama 3.1) • Free AI • {datetime.now().strftime('%H:%M:%S UTC')}_"
        
        url = f"https://api.telegram.org/bot{self.config['telegram_bot_token']}/sendMessage"
        payload = {
            'chat_id': self.config['telegram_chat_id'],
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram alert failed: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


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
        logger.error("❌ groq_api_key not found in config.json")
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
    logger.info("🎉 GROQ-POWERED AI TRADING AGENT")
    logger.info("=" * 60)
    logger.info(f"✅ 100% FREE - No credit card, no limits!")
    logger.info(f"✅ Powered by Llama 3.1 (70B parameters)")
    logger.info(f"✅ Lightning-fast inference")
    logger.info("=" * 60)
    
    # Run with auto-reconnect
    while True:
        try:
            await agent.connect()
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️  Connection closed. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down agent...")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Agent stopped by user")
        