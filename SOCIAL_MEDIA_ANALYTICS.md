# 🚀 Social Media Analytics Pipeline - Complete Beginner's Guide
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **A complete, beginner-friendly project that collects data from Reddit, Twitter, and YouTube, analyzes sentiment, and displays beautiful interactive dashboards.**

This project is perfect for learning about data science, APIs, databases, and web development all in one place!

---

## 📖 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [What You'll Learn](#-what-youll-learn)
- [Prerequisites](#-prerequisites)
- [Complete Installation Guide](#-complete-installation-guide)
- [Getting Your API Keys](#-getting-your-api-keys)
- [Project Structure Explained](#-project-structure-explained)
- [Running the Project](#-running-the-project)
- [Understanding the Code](#-understanding-the-code)
- [Troubleshooting](#-troubleshooting)
- [Making It Your Own](#-making-it-your-own)
- [FAQ](#-faq)

---

## 🎯 What This Project Does

Imagine you want to know what people are saying about "Python programming" across social media. This project:

1. **Collects** posts from Reddit, tweets from Twitter, and comments from YouTube
2. **Cleans** the text (removes URLs, special characters, etc.)
3. **Analyzes** sentiment (positive, negative, or neutral)
4. **Stores** everything in a database
5. **Visualizes** the data in a beautiful interactive dashboard

**Example Output:**
```
📊 Total collected: 150 posts
   - 50 from Reddit
   - 50 from Twitter  
   - 50 from YouTube

🔍 Sentiment Analysis:
   - Positive: 45%
   - Neutral: 35%
   - Negative: 20%

✅ All data stored in database!
```

---

## 🎓 What You'll Learn

By completing this project, you'll understand:

- ✅ **API Integration** - How to connect to and fetch data from social media platforms
- ✅ **Data Processing** - Cleaning and transforming messy real-world data
- ✅ **Natural Language Processing (NLP)** - Analyzing text to understand sentiment
- ✅ **Database Management** - Storing and querying data with SQLite
- ✅ **Data Visualization** - Creating interactive dashboards with Plotly Dash
- ✅ **Automation** - Scheduling tasks to run automatically
- ✅ **Virtual Environments** - Managing Python dependencies properly
- ✅ **Environment Variables** - Keeping API keys secure

---

## 📋 Prerequisites

### What You Need to Know

- **Basic Python** (variables, functions, loops)
- **Basic Terminal/Command Line** (cd, ls, running commands)
- **Curiosity and patience!** 

**Don't worry if you're not an expert!** This guide explains everything step-by-step.

### System Requirements

- **Ubuntu** (20.04 or newer) or Ubuntu-based Linux distro
- **Python 3.8+** (usually pre-installed)
- **Internet connection**
- **~500MB free disk space**

---

## 🛠️ Complete Installation Guide

### Step 1: Prepare Your Ubuntu System

Open Terminal with `Ctrl + Alt + T` and run these commands:

```bash
# Update your system packages
sudo apt update
sudo apt upgrade -y
```

**What this does:** Makes sure your Ubuntu has the latest security updates and software versions.

---

### Step 2: Install Python and Required Tools

```bash
# Install Python, pip (package manager), venv (virtual environments), and git
sudo apt install -y python3 python3-pip python3-venv git
```

**What this does:**
- `python3` - The Python programming language
- `python3-pip` - Tool to install Python packages
- `python3-venv` - Creates isolated Python environments
- `git` - Version control (useful for later)

**Verify installation:**
```bash
python3 --version  # Should show Python 3.8 or higher
pip3 --version     # Should show pip version
```

---

### Step 3: Create Your Project Folder

```bash
# Create a folder for the project
mkdir -p ~/social-media-analytics

# Move into that folder
cd ~/social-media-analytics
```

**What this does:** Creates a dedicated folder in your home directory for all project files.

---

### Step 4: Create a Virtual Environment

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate it
source venv/bin/activate
```

**What this does:** Creates an isolated Python environment so packages for this project don't interfere with your system Python.

**You'll know it worked** when you see `(venv)` at the start of your terminal prompt:
```
(venv) username@computer:~/social-media-analytics$
```

⚠️ **IMPORTANT:** You must run `source venv/bin/activate` every time you open a new terminal to work on this project!

---

### Step 5: Install Python Packages

Now install all the libraries we need:

```bash
# First, upgrade pip itself
pip install --upgrade pip

# Install all required packages (this may take 2-5 minutes)
pip install pandas numpy matplotlib seaborn praw tweepy google-api-python-client textblob vaderSentiment sqlalchemy plotly dash schedule python-dotenv spacy scikit-learn nltk joblib gunicorn werkzeug
```

**What each package does:**

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib`, `seaborn` | Data visualization |
| `praw` | Reddit API wrapper |
| `tweepy` | Twitter API wrapper |
| `google-api-python-client` | YouTube API wrapper |
| `textblob`, `vaderSentiment` | Sentiment analysis |
| `sqlalchemy` | Database toolkit |
| `plotly`, `dash` | Interactive dashboards |
| `schedule` | Task scheduling |
| `python-dotenv` | Load environment variables |
| `spacy` | Advanced NLP |
| `scikit-learn` | Machine learning |
| `nltk` | Natural language toolkit |
| `joblib` | Save/load models |
| `gunicorn` | Web server |
| `werkzeug` | Web utilities |

---

### Step 6: Download NLP Models

```bash
# Download Spacy's English model
python3 -m spacy download en_core_web_sm

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

**What this does:** Downloads pre-trained models needed for natural language processing (understanding text).

---

## 🔑 Getting Your API Keys

To collect data from social media platforms, you need API keys. Think of these as special passwords that let your code access their data.

### 🔴 Reddit API (5 minutes)

1. Go to: https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App" at the bottom
3. Fill in the form:
   - **Name:** `My Social Analytics App`
   - **App type:** Select `script`
   - **Description:** `Learning data science`
   - **About URL:** Leave blank
   - **Redirect URI:** `http://localhost:8080`
4. Click "Create app"
5. You'll see:
   - **Client ID** (under the app name, looks like: `xXxXxXxXxXxXx`)
   - **Secret** (next to "secret:", looks like: `yYyYyYyYyYyYyYyYyYyYy`)

**Save these for later!**

---

### 🔵 Twitter API (15 minutes)

1. Go to: https://developer.twitter.com
2. Click "Sign up" (or "Sign in" if you have an account)
3. Apply for a developer account:
   - Choose "Hobbyist" → "Exploring the API"
   - Fill in the required information
   - Agree to terms
4. Once approved, go to the Developer Portal
5. Create a Project and App:
   - Click "Create Project"
   - Name it `Social Analytics`
   - Choose "Exploring" as use case
   - Provide a description
6. In your app settings, go to "Keys and tokens"
7. Generate a **Bearer Token**

**Save this Bearer Token!**

⚠️ **Note:** Twitter approval can take a few minutes to a few hours.

---

### 🔴 YouTube API (10 minutes)

1. Go to: https://console.cloud.google.com
2. Create a new project:
   - Click "Select a project" at the top
   - Click "New Project"
   - Name it `Social Analytics`
   - Click "Create"
3. Enable YouTube Data API:
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click on it and click "Enable"
4. Create credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy your API key

**Save this API Key!**

---

### 📝 Create Your .env File

Now create a file to store all your API keys securely:

```bash
nano .env
```

Paste this template and **replace with your actual keys**:

```env
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=MyAnalyticsApp/1.0

# Twitter/X API Credentials
TWITTER_BEARER_TOKEN=your_bearer_token_here

# YouTube API Credentials
YOUTUBE_API_KEY=your_google_api_key_here
```

**Example (don't use these, they're fake!):**
```env
REDDIT_CLIENT_ID=xXxXxXxXxXxXx
REDDIT_CLIENT_SECRET=yYyYyYyYyYyYyYyYyYyYy
REDDIT_USER_AGENT=MyAnalyticsApp/1.0

TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAAAAA%2FAAAAAAAAA%3DAAAAAAAAAAAAA
YOUTUBE_API_KEY=AIzaSyBXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**Save the file:** Press `Ctrl+O`, then `Enter`, then `Ctrl+X`

⚠️ **NEVER share your .env file or push it to GitHub!**

---

## 📁 Project Structure Explained

Here's what each file in your project does:

```
social-media-analytics/
│
├── venv/                          # Virtual environment (don't touch)
│
├── .env                           # Your API keys (SECRET!)
│
├── database.py                    # Handles SQLite database operations
├── collect_data.py                # Collects posts from Reddit
├── collect_twitter.py             # Collects tweets from Twitter
├── collect_youtube.py             # Collects comments from YouTube
├── process_data.py                # Cleans and processes text data
├── sentiment_analysis.py          # Analyzes sentiment of posts
├── pipeline.py                    # Main script - runs everything
├── dashboard.py                   # Interactive web dashboard
│
├── requirements.txt               # List of all Python packages
├── .gitignore                     # Files to ignore in git
│
└── social_media.db               # SQLite database (created after first run)
```

---

## 🏗️ Creating Your Project Files

Now let's create each file with the actual code!

### File 1: `database.py` - Database Manager

```bash
nano database.py
```

Copy and paste this code:

```python
import sqlite3
import pandas as pd
from datetime import datetime

class SocialMediaDB:
    """
    This class handles all database operations.
    It creates tables and stores/retrieves posts.
    """
    
    def __init__(self, db_name='social_media.db'):
        """Initialize database connection"""
        self.db_name = db_name
        self.create_tables()
    
    def create_tables(self):
        """Create the posts table if it doesn't exist"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # SQL command to create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                title TEXT,
                text TEXT,
                author TEXT,
                created_utc TIMESTAMP,
                score INTEGER,
                num_comments INTEGER,
                url TEXT,
                sentiment_score REAL,
                sentiment_label TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Database tables created/verified!")
    
    def insert_posts(self, df):
        """Insert new posts into the database"""
        if df.empty:
            print("⚠ No data to insert")
            return
            
        conn = sqlite3.connect(self.db_name)
        
        try:
            # Insert data into table
            df.to_sql('posts', conn, if_exists='append', index=False)
            
            # Remove any duplicate posts (same ID)
            conn.execute('''
                DELETE FROM posts
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM posts
                    GROUP BY id
                )
            ''')
            conn.commit()
            print(f"✓ Inserted/Updated {len(df)} posts in database")
        except Exception as e:
            print(f"✗ Error inserting posts: {e}")
        finally:
            conn.close()
    
    def get_all_posts(self):
        """Retrieve all posts from the database"""
        conn = sqlite3.connect(self.db_name)
        try:
            df = pd.read_sql_query("SELECT * FROM posts", conn)
        except Exception as e:
            print(f"✗ Error reading posts: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()
        return df
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Creates a SQLite database (like a simple Excel file, but better)
- Defines a table structure to store posts
- Provides functions to add and retrieve posts

---

### File 2: `collect_data.py` - Reddit Collector

```bash
nano collect_data.py
```

```python
import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Try to connect to Reddit API
try:
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
    # Test if connection works
    reddit.user.me()
    print("✓ Reddit API connected successfully!")
except Exception as e:
    print(f"✗ Reddit API error: {e}")
    print("Check your .env file and Reddit API credentials")
    reddit = None

def collect_reddit_posts(subreddit_name, limit=100):
    """
    Collect posts from a specific subreddit.
    
    Args:
        subreddit_name: The subreddit to collect from (e.g., 'python')
        limit: Maximum number of posts to collect
    
    Returns:
        DataFrame with post data
    """
    if not reddit:
        print("✗ Reddit client not initialized")
        return pd.DataFrame()
    
    print(f"📥 Collecting posts from r/{subreddit_name}...")
    posts_data = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Get hot posts from the subreddit
        for post in subreddit.hot(limit=limit):
            posts_data.append({
                'id': post.id,
                'platform': 'reddit',
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'author': str(post.author),
                'url': post.url
            })
    except Exception as e:
        print(f"✗ Error collecting Reddit posts: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(posts_data)
    print(f"✓ Collected {len(df)} Reddit posts")
    return df

# This runs if you execute this file directly
if __name__ == "__main__":
    df = collect_reddit_posts('python', limit=50)
    if not df.empty:
        df.to_csv('reddit_data.csv', index=False)
        print("✓ Saved to reddit_data.csv")
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Connects to Reddit using your API keys
- Fetches posts from a subreddit (like r/python)
- Organizes the data into a nice table format

---

### File 3: `collect_twitter.py` - Twitter Collector

```bash
nano collect_twitter.py
```

```python
import tweepy
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Try to connect to Twitter API
try:
    client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))
    print("✓ Twitter API connected successfully!")
except Exception as e:
    print(f"✗ Twitter API error: {e}")
    print("Check your .env file and Twitter Bearer Token")
    client = None

def collect_twitter_tweets(query, limit=100):
    """
    Collect recent tweets matching a search query.
    
    Args:
        query: What to search for (e.g., 'python programming')
        limit: Maximum number of tweets to collect
    
    Returns:
        DataFrame with tweet data
    """
    if not client:
        print("✗ Twitter client not initialized")
        return pd.DataFrame()
    
    print(f"📥 Collecting tweets for: '{query}'...")
    tweets_data = []
    
    try:
        # Search for recent tweets
        response = client.search_recent_tweets(
            query=query,
            max_results=min(limit, 100),  # Twitter API limit is 100
            tweet_fields=['created_at', 'public_metrics', 'author_id']
        )
        
        if response.data:
            for tweet in response.data:
                tweets_data.append({
                    'id': str(tweet.id),
                    'platform': 'twitter',
                    'title': '',  # Twitter doesn't have titles
                    'text': tweet.text,
                    'author': str(tweet.author_id),
                    'created_utc': tweet.created_at,
                    'score': tweet.public_metrics['like_count'],
                    'num_comments': tweet.public_metrics['reply_count'],
                    'url': f"https://twitter.com/i/status/{tweet.id}"
                })
        else:
            print("⚠ No tweets found for that query")
            
    except Exception as e:
        print(f"✗ Error collecting tweets: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(tweets_data)
    print(f"✓ Collected {len(df)} tweets")
    return df

if __name__ == "__main__":
    df = collect_twitter_tweets('python programming', limit=50)
    if not df.empty:
        df.to_csv('twitter_data.csv', index=False)
        print("✓ Saved to twitter_data.csv")
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Connects to Twitter using your Bearer Token
- Searches for tweets matching a keyword
- Collects tweet text, likes, replies, etc.

---

### File 4: `collect_youtube.py` - YouTube Collector

```bash
nano collect_youtube.py
```

```python
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load API keys
load_dotenv()

def collect_youtube_comments(query, limit=100):
    """
    Collect comments from YouTube videos matching a search query.
    
    Args:
        query: What to search for (e.g., 'python tutorial')
        limit: Maximum number of comments to collect
    
    Returns:
        DataFrame with comment data
    """
    print(f"📥 Collecting YouTube comments for: '{query}'...")
    
    # Build YouTube API client
    try:
        youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
        print("✓ YouTube API connected successfully!")
    except Exception as e:
        print(f"✗ YouTube API error: {e}")
        print("Check your .env file and YouTube API key")
        return pd.DataFrame()
    
    # Step 1: Search for a video matching the query
    try:
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=1,  # Just get the top video
            type='video'
        ).execute()
    except Exception as e:
        print(f"✗ Error searching videos: {e}")
        return pd.DataFrame()
    
    if not search_response.get('items'):
        print("✗ No videos found for that query")
        return pd.DataFrame()
    
    video_id = search_response['items'][0]['id']['videoId']
    video_title = search_response['items'][0]['snippet']['title']
    print(f"📹 Found video: '{video_title}'")
    
    # Step 2: Get comments from that video
    comments_data = []
    try:
        comment_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=min(limit, 100)
        ).execute()
        
        for item in comment_response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments_data.append({
                'id': item['id'],
                'platform': 'youtube',
                'title': '',
                'text': comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'created_utc': datetime.strptime(comment['publishedAt'], "%Y-%m-%dT%H:%M:%SZ"),
                'score': comment['likeCount'],
                'num_comments': item['snippet']['totalReplyCount'],
                'url': f"https://youtube.com/watch?v={video_id}&lc={item['id']}"
            })
    except Exception as e:
        print(f"✗ Error getting comments (may be disabled): {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(comments_data)
    print(f"✓ Collected {len(df)} comments")
    return df

if __name__ == "__main__":
    df = collect_youtube_comments('python tutorial', limit=50)
    if not df.empty:
        df.to_csv('youtube_data.csv', index=False)
        print("✓ Saved to youtube_data.csv")
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Connects to YouTube using your API key
- Searches for a video matching your query
- Collects comments from that video

---

### File 5: `process_data.py` - Data Cleaner

```bash
nano process_data.py
```

```python
import pandas as pd
import re

def clean_text(text):
    """
    Clean text data by removing URLs, special characters, and extra spaces.
    
    Args:
        text: The text to clean
    
    Returns:
        Cleaned text in lowercase
    """
    # Handle empty or non-string values
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs (like http://example.com)
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase
    return text.lower()

def process_data(df):
    """
    Process and clean a DataFrame of posts.
    
    Args:
        df: DataFrame with raw post data
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
        
    print(f"⚙ Processing {len(df)} posts...")
    
    # Make sure 'title' column exists (Twitter/YouTube don't have titles)
    if 'title' not in df.columns:
        df['title'] = ''
    
    # Clean title and text
    df['title_clean'] = df['title'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Combine title and text into one column
    df['full_text'] = (df['title_clean'] + ' ' + df['text_clean']).str.strip()
    
    # Remove duplicate posts
    df = df.drop_duplicates(subset=['id'])
    
    # Remove posts with very short text (less than 10 characters)
    df = df[df['full_text'].str.len() > 10]
    
    print(f"✓ {len(df)} posts remaining after processing")
    return df
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Cleans messy text data
- Removes URLs, emojis, special characters
- Combines title and text into one field for analysis

---

### File 6: `sentiment_analysis.py` - Sentiment Analyzer

```bash
nano sentiment_analysis.py
```

```python
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# Try to load custom model (if you train one later)
try:
    custom_model = joblib.load('custom_sentiment_model.pkl')
    print("✓ Custom sentiment model loaded")
except FileNotFoundError:
    print("⚠ Custom model not found, using VADER only")
    custom_model = None

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    VADER is great for social media text!
    
    Args:
        text: The text to analyze
    
    Returns:
        (score, label) tuple
        - score: -1 (very negative) to +1 (very positive)
        - label: 'positive', 'negative', or 'neutral'
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    # Get compound score (overall sentiment)
    compound = scores['compound']
    
    # Classify sentiment
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    return compound, label

def analyze_data(df):
    """
    Analyze sentiment for all posts in a DataFrame.
    
    Args:
        df: DataFrame with 'full_text' column
    
    Returns:
        DataFrame with sentiment_score and sentiment_label columns added
    """
    if df.empty:
        return df
        
    print("🔍 Analyzing sentiment...")
    
    # Apply sentiment analysis to each post
    df[['sentiment_score', 'sentiment_label']] = df['full_text'].apply(
        lambda x: pd.Series(analyze_sentiment_vader(x))
    )
    
    print("✓ Sentiment analysis complete!")
    print("\n📊 Sentiment Distribution:")
    print(df['sentiment_label'].value_counts())
    print()
    
    return df
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Uses VADER to analyze sentiment (how positive/negative text is)
- Assigns a score from -1 (very negative) to +1 (very positive)
- Labels each post as positive, negative, or neutral

---

### File 7: `pipeline.py` - Main Pipeline (THE BRAIN!)

```bash
nano pipeline.py
```

```python
import schedule
import time
import pandas as pd
from collect_data import collect_reddit_posts
from collect_twitter import collect_twitter_tweets
from collect_youtube import collect_youtube_comments
from process_data import process_data
from sentiment_analysis import analyze_data
from database import SocialMediaDB

def run_pipeline():
    """
    This is the main function that runs everything:
    1. Collects data from all platforms
    2. Processes and cleans the data
    3. Analyzes sentiment
    4. Stores everything in the database
    """
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Run the pipeline immediately when script starts
    run_pipeline()
    
    # Schedule to run automatically every 6 hours
    schedule.every(6).hours.do(run_pipeline)
    
    print("\n⏰ Pipeline scheduled to run every 6 hours")
    print("Press Ctrl+C to stop\n")
    
    # Keep the script running and check for scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
    print("🚀 STARTING PIPELINE RUN")
    print("="*60)
    
    # ====== STEP 1: COLLECT DATA ======
    print("\n📥 Step 1: Collecting data from all platforms...")
    
    df_reddit = collect_reddit_posts('datascience', limit=50)
    df_twitter = collect_twitter_tweets('data engineering', limit=50)
    df_youtube = collect_youtube_comments('machine learning', limit=50)
    
    # Combine all data into one DataFrame
    df_all = pd.concat([df_reddit, df_twitter, df_youtube], ignore_index=True)
    
    print(f"\n📊 Total collected: {len(df_all)} posts")
    print(f"   - Reddit: {len(df_reddit)} posts")
    print(f"   - Twitter: {len(df_twitter)} tweets")
    print(f"   - YouTube: {len(df_youtube)} comments")
    
    if df_all.empty:
        print("⚠ No data collected. Exiting...")
        return
    
    # ====== STEP 2: PROCESS DATA ======
    print("\n⚙ Step 2: Processing and cleaning data...")
    df_processed = process_data(df_all)
    
    if df_processed.empty:
        print("⚠ No data after processing. Exiting...")
        return
    
    # ====== STEP 3: ANALYZE SENTIMENT ======
    print("\n🔍 Step 3: Analyzing sentiment...")
    df_analyzed = analyze_data(df_processed)
    
    # ====== STEP 4: STORE IN DATABASE ======
    print("\n💾 Step 4: Storing in database...")
    
    # Select only the columns we want in the database
    db_columns = [
        'id', 'platform', 'title', 'text', 'author', 'created_utc',
        'score', 'num_comments', 'url', 'sentiment_score', 'sentiment_label'
    ]
    
    df_to_insert = pd.DataFrame()
    for col in db_columns:
        df_to_insert[col] = df_analyzed[col] if col in df_analyzed.columns else None
    
    db = SocialMediaDB()
    db.insert_posts(df_to_insert)
    
    print("\n" + "="*60)
