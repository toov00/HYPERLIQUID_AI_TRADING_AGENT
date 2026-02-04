# Hyperliquid AI Trading Agent

An AI-powered trading agent that uses Groq's free API with Llama 3.1 to analyze Hyperliquid market conditions and send intelligent trading alerts based on real-time data.

<img src="assets/interface.png" alt="Discord Interface">

## What It Does

Monitors Hyperliquid order books and trade flow in real-time, analyzes market conditions using Groq's free AI API, and sends personalized trading alerts when opportunities match your trading style and risk profile.

**Features:**
- Real-time market monitoring via WebSocket
- AI-powered analysis using Groq's free API (Llama 3.1)
- No API costs, completely free to use
- Personalized recommendations based on your trading style
- Discord alert integration
- Opportunity scoring (0-10 scale)
- Execution strategy recommendations
- Risk factor identification

## Installation

**Requirements:** Python 3.8+

1. Clone the repository:

```bash
git clone <your-repo>
cd HYPER_AGENT
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- `websockets`: WebSocket connection to Hyperliquid
- `discord-webhook`: Discord alert integration
- `requests`: HTTP requests

Get your free Groq API key:

1. Go to https://console.groq.com/
2. Sign up for a free account
3. Navigate to API Keys
4. Create a new API key
5. Copy your key (starts with `gsk_...`)

Add it to your config file (see Configuration below).

## Usage

### Quick Start

1. Copy the example config:

```bash
cp config.groq.json config.json
```

2. Edit `config.json` with your settings:

```json
{
  "coin": "HYPE",
  "groq_api_key": "gsk-your-key-here",
  "groq_model": "llama-3.1-8b-instant",
  "ai_analysis_interval": 60,
  "alert_cooldown": 300,
  "user_preferences": {
    "trading_style": "balanced",
    "risk_tolerance": "medium",
    "position_size": "medium",
    "time_horizon": "intraday"
  },
  "discord_webhook": "https://discord.com/api/webhooks/..."
}
```

3. Run the agent:

```bash
python hyperliquid_groq_agent.py
```

### Configuration

1. Set Trading Preferences:

```json
{
  "user_preferences": {
    "trading_style": "balanced",
    "risk_tolerance": "medium",
    "position_size": "medium",
    "time_horizon": "intraday"
  }
}
```

Trading styles: `conservative`, `balanced`, `aggressive`
Risk tolerance: `low`, `medium`, `high`
Position size: `small`, `medium`, `large`
Time horizon: `scalping`, `intraday`, `swing`

2. Configure Groq Model:

The default model is `llama-3.1-8b-instant`. You can also use:
- `llama-3-70b-8192`: Larger, more capable model
- `mixtral-8x7b-32768`: Alternative model option

All models are free to use with Groq's API.

3. Configure Discord Webhook:

Add your Discord webhook URL to receive alerts when the AI detects trading opportunities.

## Trading Signals

The agent generates five types of signals:

1. **STRONG BUY SIGNAL**: Excellent entry conditions (score 8-10)
2. **BUY SIGNAL**: Good entry conditions (score 7-8)
3. **NEUTRAL**: Wait for better conditions (score 4-6)
4. **SELL SIGNAL**: Good exit conditions (score 7-8)
5. **STRONG SELL SIGNAL**: Excellent exit conditions (score 8-10)

Each signal includes:
- Opportunity score (0-10)
- Market assessment
- Reasoning with specific factors
- Execution strategy recommendations
- Risk factors to watch

## Examples

See the main script `hyperliquid_groq_agent.py` for the complete implementation. The agent:

- Connects to Hyperliquid WebSocket
- Monitors order book depth and spreads
- Tracks recent trade flow
- Analyzes conditions every 60 seconds (configurable) using Groq's free API
- Sends alerts when opportunity score >= 7

## Alerts

Alerts are sent via Discord webhooks when the AI detects favorable trading conditions. Each alert includes:

- Current market price and spread
- Opportunity score
- AI reasoning with key factors
- Execution strategy (market vs limit orders, slippage estimates)
- Risk factors to consider

Alerts respect the cooldown period (default 300 seconds) to avoid spam.

## Troubleshooting

**Connection errors?** Check your internet connection and verify Hyperliquid API is accessible.

**No alerts being sent?** The AI may not see favorable conditions. Check logs for opportunity scores, or adjust your preferences to be more aggressive.

**API rate limits?** Groq's free tier is very generous, but if you hit limits, increase `ai_analysis_interval` to reduce API calls.

**Import errors?** Make sure all dependencies are installed: `pip install -r requirements.txt`

## Contributing

Contributions welcome! To add new features:
1. Fork the repository
2. Add your feature or improvement
3. Test with different market conditions
4. Submit a pull request

## License

MIT License

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and never trade more than you can afford to lose. Past performance does not guarantee future results.

## Resources

- [Hyperliquid Documentation](https://hyperliquid.gitbook.io/hyperliquid-docs)
- [Groq API Documentation](https://console.groq.com/docs)
- [Discord Webhooks Guide](https://discord.com/developers/docs/resources/webhook)
