import discord
from discord import app_commands
import os
from dotenv import load_dotenv
from news import fetch_news
from discord import Embed
from discord.ext import tasks
from itertools import cycle
import asyncio
from datetime import datetime, timedelta



# Category rotation system
CATEGORIES = ["technology", "science", "business", "health"]
current_category_index = 0

def get_next_category():
    """Get the next category in rotation"""
    global current_category_index
    category = CATEGORIES[current_category_index]
    current_category_index = (current_category_index + 1) % len(CATEGORIES)
    return category

def peek_next_category():
    """Peek at the next category without advancing"""
    return CATEGORIES[current_category_index]

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("DISCORD_GUILD_ID"))  # Convert string to integer
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

intents = discord.Intents.default()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)




@tasks.loop(minutes=5)
async def auto_news():
    """Auto-send news updates every 5 minutes"""
    print(f"🔄 Auto-news triggered at {datetime.now()}")
    
    # Get the channel (make sure this is a CHANNEL ID, not server ID)
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print(f"❌ Channel {CHANNEL_ID} not found!")
        return
    
    current_category = get_next_category()
    print(f"📰 Fetching {current_category} news...")
    
    try:
        articles = await fetch_news([current_category], max_per_category=3)
        if not articles:
            print("⚠️ No articles found")
            return
        
        print(f"✅ Found {len(articles)} articles")
        
        # Time-based greeting
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            greeting = "🌞 Morning"
        elif 12 <= current_hour < 17:
            greeting = "🌇 Afternoon"
        else:
            greeting = "🌙 Evening"
        
        embed = discord.Embed(
            title=f"{greeting} {current_category.upper()} News Update",
            color=0x7289da,
            timestamp=datetime.now()
        )
        
        for article in articles:
            date_str = article["raw_date"][:10] if article["raw_date"] else "Date unknown"
            title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']
            tags_str = ", ".join(article["tags"]) if article["tags"] else "No tags"
            
            embed.add_field(
                name=title,
                value=(
                    f"📅 {date_str} | {article['source']}\n"
                    f"🔹 {article['summary']}\n"
                    f"🏷️ Tags: {tags_str}\n"
                    f"[Read more]({article['url']})"
                ),
                inline=False
            )
        
        embed.set_footer(text=f"Auto-update • Next: {peek_next_category()} at {datetime.now() + timedelta(minutes=5):%H:%M}")
        await channel.send(embed=embed)
        print("✅ Auto-news sent successfully!")
        
    except Exception as e:
        print(f"❌ Auto-news error: {e}")
        # Send error message to channel for debugging
        try:
            await channel.send(f"⚠️ Auto-news failed: {str(e)}")
        except:
            pass

@tree.command(guild=discord.Object(id=GUILD_ID), name="test-auto", description="Test auto-news manually")
async def test_auto(interaction: discord.Interaction):
    await interaction.response.defer()
    
    # Override the channel for testing to use the interaction channel
    global CHANNEL_ID
    original_channel = CHANNEL_ID
    CHANNEL_ID = interaction.channel.id
    
    print(f"🧪 Testing auto-news in channel {interaction.channel.id}")
    
    try:
        await auto_news()
        await interaction.followup.send("✅ Auto-news test completed! Check above for the news update.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"❌ Test failed: {str(e)}", ephemeral=True)
        print(f"Test error: {e}")
    finally:
        # Restore original channel
        CHANNEL_ID = original_channel

@bot.event
async def on_ready():
    print(f"✅ Bot ready as {bot.user}")
    print(f"📺 Target channel ID: {CHANNEL_ID}")
    
    # Verify channel exists
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        print(f"✅ Found target channel: #{channel.name}")
    else:
        print(f"❌ WARNING: Channel {CHANNEL_ID} not found!")
    
    # Sync commands to server
    try:
        await tree.sync(guild=discord.Object(id=GUILD_ID))
        print("✅ Commands synced to server!")
    except Exception as e:
        print(f"❌ Sync failed: {e}")
    
    # Start auto-news task
    if not auto_news.is_running():
        auto_news.start()
        print("✅ Auto-news task started!")

@auto_news.before_loop
async def before_auto():
    await bot.wait_until_ready()
    print(f"⏰ Auto-news task ready. Next run in 5 minutes.")

@auto_news.error
async def auto_news_error(error):
    print(f"❌ Auto-news task error: {error}")

@tree.command(guild=discord.Object(id=GUILD_ID), name="hello", description="Greet user")
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message(f"👋 Hello {interaction.user.mention}!")

@tree.command(guild=discord.Object(id=GUILD_ID), name="server", description="Show server info")
async def server(interaction: discord.Interaction):
    guild = interaction.guild
    created_at = guild.created_at.strftime("%B %d, %Y")
    
    embed = discord.Embed(
        title=f"📊 {guild.name} Info",
        color=discord.Color.blue()
    )
    embed.add_field(name="👥 Members", value=guild.member_count)
    embed.add_field(name="🆔 Server ID", value=guild.id)
    embed.add_field(name="📅 Created On", value=created_at)
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    
    await interaction.response.send_message(embed=embed)

@tree.command(guild=discord.Object(id=GUILD_ID), name="news", description="Get latest headlines")
async def news(
    interaction: discord.Interaction,
    categories: str = "technology",
    count: int = 3
):
    await interaction.response.defer()
    
    try:
        count = max(1, min(10, count))
        topics = [t.strip() for t in categories.split(",")]
        articles = await fetch_news(topics, max_per_category=count)
        
        if not articles:
            return await interaction.followup.send("⚠️ No articles found. Try different categories.")
        
        articles.sort(key=lambda x: x["date"], reverse=True)
        embed = discord.Embed(
            title=f"📰 Latest News ({count} articles)",
            color=0x00ff00
        )
        
        for article in articles[:count]:
            date_str = article["raw_date"][:10] if article["raw_date"] else "Date unknown"
            title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']
            tags_str = ", ".join(article["tags"]) if article["tags"] else "No tags"
            
            embed.add_field(
                name=f"{article['category']}: {title}",
                value=(
                    f"📅 {date_str} | {article['source']}\n"
                    f"🔹 **Summary**: {article['summary']}\n"
                    f"🏷️ **Tags**: {tags_str}\n"
                    f"[Read more]({article['url']})"
                ),
                inline=False
            )
            
        await interaction.followup.send(embed=embed)
        
    except ValueError as e:
        await interaction.followup.send(f"❌ {str(e)}")
    except Exception as e:
        await interaction.followup.send("⚠️ Failed to fetch news")
        print(f"News Error: {e}")




@tree.command(guild=discord.Object(id=GUILD_ID), name="debug", description="Check bot status")
async def debug(interaction: discord.Interaction):
    channel = bot.get_channel(CHANNEL_ID)
    
    embed = discord.Embed(title="🔍 Bot Debug Info", color=0xff9900)
    embed.add_field(name="Bot User", value=f"{bot.user} (ID: {bot.user.id})")
    embed.add_field(name="Target Channel", value=f"#{channel.name} (ID: {CHANNEL_ID})" if channel else f"❌ Channel {CHANNEL_ID} not found")
    embed.add_field(name="Auto-news Running", value="✅ Yes" if auto_news.is_running() else "❌ No")
    embed.add_field(name="Next Category", value=peek_next_category())  # Fixed this line
    
    await interaction.response.send_message(embed=embed, ephemeral=True)
    


# --------- DEMO MODE (no Discord) ----------
async def _demo(categories: str = "technology", count: int = 3):
    topics = [t.strip() for t in categories.split(",")]
    articles = await fetch_news(topics, max_per_category=count)
    if not articles:
        print("No articles found.")
        return
    articles.sort(key=lambda x: x["date"], reverse=True)
    print(f"\n=== Demo: {count} summarized articles ===")
    for a in articles[:count]:
        date_str = (a.get("raw_date") or "")[:10] or "Date unknown"
        tags_str = ", ".join(a["tags"]) if a["tags"] else "No tags"
        print(f"\n[{a['category']}] {a['title']} ({a['source']}, {date_str})")
        print(f"Summary: {a['summary']}")
        print(f"Tags: {tags_str}")
        print(f"Link: {a['url']}")

if __name__ == "__main__":
    import sys, asyncio, os
    from dotenv import load_dotenv
    load_dotenv()

    # Read IDs from env if you want to avoid hardcoding
    try:
        GUILD_ID = int(os.getenv("DISCORD_GUILD_ID")) if os.getenv("DISCORD_GUILD_ID") else None
        CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID")) if os.getenv("DISCORD_CHANNEL_ID") else None
    except ValueError:
        GUILD_ID = CHANNEL_ID = None

    TOKEN = os.getenv("DISCORD_TOKEN")

    # Demo mode for reviewers
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        cats = sys.argv[2] if len(sys.argv) > 2 else "technology"
        try:
            cnt = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        except ValueError:
            cnt = 3
        asyncio.run(_demo(cats, cnt))
        raise SystemExit(0)

    # Only run the actual bot if a token is present
    if not TOKEN:
        print("No DISCORD_TOKEN found. Try demo mode:\n  python bot.py --demo \"technology,science\" 3")
        raise SystemExit(0)
    
    bot.run(TOKEN)

