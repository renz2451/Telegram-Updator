# Telegram Offset Updater Bot

A Telegram bot that automatically updates offsets in C++ files using .cs dump files.

## Features
- 📤 Upload C++ source files and .cs dump files via Telegram
- 🔄 Automatically parse .cs dump files for function addresses
- ⚡ Update offsets with comments showing old values
- 📁 Send back updated files and detailed logs
- 🔒 Secure - Only authorized users can access

## Setup

### 1. Create Telegram Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot`
3. Choose a name for your bot
4. Get the API token

### 2. Get Your User ID
1. Message [@userinfobot](https://t.me/userinfobot)
2. Copy your numeric user ID

### 3. Configure Bot
Edit `telegram_bot.py`:
```python
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
ALLOWED_USER_IDS = [YOUR_USER_ID_HERE]