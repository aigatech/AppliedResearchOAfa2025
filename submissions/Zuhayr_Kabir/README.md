# Discord News Bot with AI Summarization

## What it does
This Discord bot fetches news articles from various categories, generates AI-powered summaries using HuggingFace's DistilBART model, and posts them to a Discord channel. It features:
- Automated news updates every 5 minutes with category rotation
- Manual news fetching commands
- AI-generated summaries of articles
- Time-based greetings
- Performs zero-shot-classification to assign tags to the different news sources

## How to run it
1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in a `.env` file (copy from `env.example`)
3. Get API keys:
   - Discord Bot Token from Discord Developer Portal
   - NewsAPI key from newsapi.org
4. Run the bot: `python bot.py`
5. For demo mode, running without Discord: `python bot.py --demo "technology" 3` or any other categories from the valid categories list in place of "technology"

## Dependencies
- discord.py
- transformers
- aiohttp
- python-dotenv
- torch