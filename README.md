# PROJECT-TO-COMPLETE

# 🚀 My Social Media Analytics Pipeline

**Objective:** A 7-day sprint to build a multi-platform (Reddit, Twitter, YouTube) analytics pipeline. This document contains all steps and all code from start to finish.

## 🧰 Project Setup (Day 0)

**Goal:** Prepare your Ubuntu system, Python environment, and API keys.

### Step 1: System Setup

Open Terminal (`Ctrl+Alt+T`) and run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

### Step 2: Python Environment

```bash
# Create project folder
mkdir ~/social-media-analytics
cd ~/social-media-analytics

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

*(You must run `source venv/bin/activate` every time you open a new terminal for this project)*

### Step 3: Install All Libraries

Run this one command to install everything you'll need:

```bash
pip install pandas numpy matplotlib seaborn praw tweepy google-api-python-client textblob vaderSentiment sqlalchemy sqlite3 plotly dash schedule python-dotenv spacy scikit-learn nltk joblib gunicorn
```

### Step 4: Download NLP Models

```bash
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Step 5: Create `.env` Secret File

This file will hold all your API keys.

```bash
nano .env
```

Paste this in and add your keys. **Get these from Day 1, 2, and 3.**

```
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=MyAnalyticsApp/1.0

# Twitter/X API Credentials
TWITTER_BEARER_TOKEN=your_bearer_token_here

# YouTube API Credentials
YOUTUBE_API_KEY=your_google_api_key_here
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 1: Reddit Collector (The Foundation)

**Goal:** Create the Reddit data collector. Get your API keys at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).

### File: `collect_data.py`

```bash
nano collect_data.py
```

Paste this code (this is your original Reddit script):

```python
import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Reddit API
try:
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
except Exception as e:
    print(f"Error initializing PRAW: {e}")
    reddit = None

def collect_reddit_posts(subreddit_name, limit=100):
    """Collect posts from a subreddit"""
    if not reddit:
        print("Reddit client not initialized. Check API credentials.")
        return pd.DataFrame()

    print(f"Collecting posts from r/{subreddit_name}...")
    
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    try:
        for post in subreddit.hot(limit=limit):
            posts_data.append({
                'id': post.id,
                'platform': 'reddit', # Added for unification
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'author': str(post.author),
                'url': post.url
            })
    except Exception as e:
        print(f"Error collecting from Reddit: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(posts_data)
    print(f"Collected {len(df)} Reddit posts!")
    return df

if __name__ == "__main__":
    # Test the function
    df = collect_reddit_posts('python', limit=50)
    
    # Save to CSV
    df.to_csv('reddit_data.csv', index=False)
    print("Reddit data saved to reddit_data.csv")
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 2: Twitter/X Collector

**Goal:** Create the Twitter data collector. Get your API key at [https://developer.twitter.com](https://developer.twitter.com).

### File: `collect_twitter.py`

```bash
nano collect_twitter.py
```

Paste this code:

```python
import tweepy
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

# Initialize Tweepy Client (for API v2)
try:
    client = tweepy.Client(BEARER_TOKEN)
except Exception as e:
    print(f"Error initializing Tweepy: {e}")
    client = None

def collect_twitter_tweets(query, limit=100):
    """Collect tweets using Twitter API v2"""
    if not client:
        print("Tweepy client not initialized. Check Bearer Token.")
        return pd.DataFrame()

    print(f"Collecting tweets for query: {query}...")
    
    if limit > 100:
        limit = 100 # API v2 basic limit per request
    
    tweets_data = []
    try:
        response = client.search_recent_tweets(
            query=query, 
            max_results=limit, 
            tweet_fields=['created_at', 'public_metrics', 'author_id']
        )
        
        if response.data:
            for tweet in response.data:
                tweets_data.append({
                    'id': tweet.id,
                    'platform': 'twitter',
                    'text': tweet.text,
                    'author': tweet.author_id, # Using author_id
                    'created_utc': tweet.created_at,
                    'score': tweet.public_metrics['like_count'], # Use likes
                    'num_comments': tweet.public_metrics['reply_count'],
                    'url': f"https://twitter.com/anyuser/status/{tweet.id}"
                })
    except Exception as e:
        print(f"Error collecting from Twitter: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(tweets_data)
    print(f"Collected {len(df)} tweets!")
    return df

if __name__ == "__main__":
    # Test the function
    df = collect_twitter_tweets('python programming', limit=50)
    df.to_csv('twitter_data.csv', index=False)
    print("Twitter data saved to twitter_data.csv")
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 3: YouTube Collector

**Goal:** Create the YouTube data collector. Get your API key from the [Google Cloud Console](https://console.cloud.google.com/).

### File: `collect_youtube.py`

```bash
nano collect_youtube.py
```

Paste this code:

```python
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def collect_youtube_comments(query, limit=100):
    """Collect video comments from YouTube"""
    print(f"Collecting YouTube comments for query: {query}...")
    
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    except Exception as e:
        print(f"Error building YouTube client: {e}")
        return pd.DataFrame()

    # 1. Search for a relevant video
    try:
        search_response = youtube.search().list(
            q=query, part='snippet', maxResults=1, type='video'
        ).execute()
    except Exception as e:
        print(f"Error searching for video: {e}")
        return pd.DataFrame()
    
    if not search_response.get('items'):
        print("No videos found for that query.")
        return pd.DataFrame()

    video_id = search_response['items'][0]['id']['videoId']
    video_title = search_response['items'][0]['snippet']['title']
    print(f"Collecting comments from video: '{video_title}'")

    # 2. Get comments from that video
    comments_data = []
    try:
        comment_response = youtube.commentThreads().list(
            part='snippet', videoId=video_id, textFormat='plainText', maxResults=limit
        ).execute()
    except Exception as e:
        print(f"Could not get comments (comments may be disabled): {e}")
        return pd.DataFrame()

    for item in comment_response.get('items', []):
        comment = item['snippet']['topLevelComment']['snippet']
        comments_data.append({
            'id': item['id'],
            'platform': 'youtube',
            'text': comment['textDisplay'],
            'author': comment['authorDisplayName'],
            'created_utc': datetime.strptime(comment['publishedAt'], "%Y-%m-%dT%H:%M:%SZ"),
            'score': comment['likeCount'],
            'num_comments': item['snippet']['totalReplyCount'],
            'url': f"https://www.youtube.com/watch?v={video_id}&lc={item['id']}"
        })

    df = pd.DataFrame(comments_data)
    print(f"Collected {len(df)} comments!")
    return df

if __name__ == "__main__":
    # Test the function
    df = collect_youtube_comments('python tutorial', limit=50)
    df.to_csv('youtube_data.csv', index=False)
    print("YouTube data saved to youtube_data.csv")
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 4: Unified Database & Processing

**Goal:** Create the database and processing scripts to handle all platforms.

### File 1: `database.py`

*(This file defines the database structure and functions)*

```bash
nano database.py
```

Paste this code:

```python
import sqlite3
import pandas as pd
from datetime import datetime

class SocialMediaDB:
    def __init__(self, db_name='social_media.db'):
        self.db_name = db_name
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Generic schema to hold data from all platforms
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
        print("Database tables created/verified!")
    
    def insert_posts(self, df):
        """Insert posts into database, replacing duplicates"""
        conn = sqlite3.connect(self.db_name)
        
        # Use 'replace' to handle duplicates based on the PRIMARY KEY (id)
        df.to_sql('posts', conn, if_exists='append', index=False)
        
        # Remove duplicates manually just in case to_sql fails
        # This is a good safeguard
        try:
            conn.execute('''
                DELETE FROM posts
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM posts
                    GROUP BY id
                )
            ''')
            conn.commit()
        except Exception as e:
            print(f"Error cleaning duplicates: {e}")

        conn.close()
        print(f"Inserted/Updated {len(df)} posts in database")

    def get_all_posts(self):
        """Retrieve all posts"""
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql_query("SELECT * FROM posts", conn)
        conn.close()
        return df
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 2: `process_data.py`

*(This file defines text cleaning functions)*

```bash
nano process_data.py
```

Paste this code:

```python
import pandas as pd
import re

def clean_text(text):
    """Clean text data"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.lower()

def process_data(df):
    """Process and clean data from any platform"""
    print(f"Processing {len(df)} posts...")
    
    # Handle optional 'title' column
    if 'title' in df.columns:
        df['title_clean'] = df['title'].apply(clean_text)
    else:
        df['title_clean'] = '' # Create empty column if it doesn't exist
        
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Combine title and text
    df['full_text'] = df['title_clean'] + ' ' + df['text_clean']
    df['full_text'] = df['full_text'].str.strip() # Remove leading/trailing spaces
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['id'])
    
    # Remove empty posts
    df = df[df['full_text'].str.len() > 10]
    
    print(f"Processed {len(df)} posts remaining")
    return df
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 5: NLP & Custom ML Model

**Goal:** Create scripts for sentiment analysis (VADER + custom model), topic modeling (LDA), and entity recognition (NER).

### File 1: `sentiment_analysis.py`

*(This file runs VADER and loads our custom model)*

```bash
nano sentiment_analysis.py
```

Paste this code:

```python
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import joblib 

# Load custom model ONCE when the script loads
try:
    custom_model = joblib.load('custom_sentiment_model.pkl')
    print("Custom sentiment model loaded successfully.")
except FileNotFoundError:
    print("Custom model 'custom_sentiment_model.pkl' not found.")
    print("Run train_model.py first. Using VADER only.")
    custom_model = None

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05: label = 'positive'
    elif compound <= -0.05: label = 'negative'
    else: label = 'neutral'
    return compound, label

def analyze_sentiment_custom(text):
    """Analyze sentiment using our trained model"""
    if not custom_model:
        return 0.0, 'neutral' # Fallback
        
    try:
        prediction = custom_model.predict([text])[0]
        probabilities = custom_model.predict_proba([text])[0]
        
        if prediction == 1:
            label = 'positive'
            score = probabilities[1] # Positive probability
        else:
            label = 'negative'
            score = -probabilities[0] # Negative probability
        return score, label
    except Exception as e:
        print(f"Error with custom model: {e}")
        return 0.0, 'neutral'

def analyze_data(df):
    """Analyze sentiment for all posts"""
    print("Analyzing sentiment...")
    
    # --- We will use VADER as our main score for the database ---
    df[['sentiment_score', 'sentiment_label']] = df['full_text'].apply(
        lambda x: pd.Series(analyze_sentiment_vader(x))
    )
    print("VADER analysis complete.")
    
    # --- We can also run our custom model to compare in the console ---
    if custom_model:
        df[['custom_score', 'custom_label']] = df['full_text'].apply(
            lambda x: pd.Series(analyze_sentiment_custom(x))
        )
        print("Custom model analysis complete.")
        print("\n--- Custom Model Distribution ---")
        print(df['custom_label'].value_counts(normalize=True))
    
    print("\n--- VADER Sentiment Distribution ---")
    print(df['sentiment_label'].value_counts(normalize=True))
    
    return df
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 2: `train_model.py`

*(This file trains your custom sentiment model. You only need to run it once after downloading the data.)*

**Action:** Download the [Sentiment140 dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140). Unzip it and rename the CSV file to `training_data.csv` in your project folder.

```bash
nano train_model.py
```

Paste this code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import re

print("Starting model training...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text) # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-letters
    text = ' '.join(text.split())
    return text

try:
    df = pd.read_csv(
        'training_data.csv', 
        encoding='latin-1', 
        header=None,
        names=['target', 'id', 'date', 'query', 'user', 'text']
    )
except FileNotFoundError:
    print("ERROR: 'training_data.csv' not found.")
    print("Download from Kaggle and rename it.")
    exit()

print("Preprocessing data...")
df = df[['target', 'text']]
df = df.dropna()
df['target'] = df['target'].replace(4, 1) # Change 4 to 1 (positive)
df = df[df['target'].isin([0, 1])]
df['target'] = df['target'].astype(int)

print("Sampling data (100,000 rows)...")
df_sample = df.sample(n=100000, random_state=42)
df_sample['text_clean'] = df_sample['text'].apply(clean_text)

X = df_sample['text_clean']
y = df_sample['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(solver='liblinear'))
])

print(f"Training model on {len(X_train)} samples...")
pipeline.fit(X_train, y_train)

print("Evaluating model...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

joblib.dump(pipeline, 'custom_sentiment_model.pkl')
print("\nModel saved as 'custom_sentiment_model.pkl'!")
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 3: `advanced_analysis.py`

*(This script runs NER and LDA. You can run this separately.)*

```bash
nano advanced_analysis.py
```

Paste this code:

```python
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import pandas as pd
from database import SocialMediaDB 

try:
    nlp = spacy.load('en_core_web_sm')
except IOError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Run: python3 -m spacy download en_core_web_sm")
    exit()
    
stop_words = list(stopwords.words('english'))
stop_words.extend(['http', 'https', 'com', 'www', 'rt', 'im', 'lol', 've', 'just'])

def extract_entities(df):
    """Extract Named Entities (NER) from full_text"""
    print("\nExtracting entities (NER)...")
    entity_counts = {}
    
    for doc in nlp.pipe(df['full_text'], batch_size=50):
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']: # People, Orgs, Places
                text = ent.text.strip().lower()
                if text and len(text) > 2:
                    entity_counts[text] = entity_counts.get(text, 0) + 1
    
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("--- Top 15 Entities ---")
    for entity, count in top_entities:
        print(f"{entity}: {count}")
    return df

def find_topics_lda(df, num_topics=5, num_words=4):
    """Find main topics using LDA"""
    print(f"\nFinding {num_topics} topics (LDA)...")
    
    # Filter out empty text
    non_empty_text = df[df['full_text'].str.len() > 0]['full_text']
    if non_empty_text.empty:
        print("No text data to analyze for LDA.")
        return

    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform(non_empty_text)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    
    print("\n--- Top Discovered Topics ---")
    for i, topic in enumerate(lda.components_):
        top_words = [feature_names[idx] for idx in topic.argsort()[-num_words:]]
        print(f"Topic {i+1}: {', '.join(top_words)}")
    return

if __name__ == "__main__":
    print("Loading data from database for analysis...")
    db = SocialMediaDB()
    df = db.get_all_posts()
    
    if not df.empty and 'full_text' in df.columns:
        extract_entities(df)
        find_topics_lda(df)
    else:
        print("Database is empty. Run pipeline.py first.")
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 6: The Master Pipeline

**Goal:** Create the main `pipeline.py` that connects and runs everything.

### File: `pipeline.py`

```bash
nano pipeline.py
```

Paste this code:

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
    """Run the entire analytics pipeline for all platforms"""
    print("\n=== Starting Full Pipeline Run ===")
    
    # 1. Collect data
    df_reddit = collect_reddit_posts('datascience', limit=50) 
    df_twitter = collect_twitter_tweets('data engineering', limit=50) 
    df_youtube = collect_youtube_comments('machine learning', limit=50) 
    
    # Combine into one DataFrame
    df_all = pd.concat([df_reddit, df_twitter, df_youtube], ignore_index=True)
    print(f"Collected a total of {len(df_all)} posts from 3 platforms.")

    if df_all.empty:
        print("No data collected. Exiting pipeline run.")
        print("=== Pipeline Complete (No Data) ===\n")
        return

    # 2. Process data
    df_processed = process_data(df_all)
    
    if df_processed.empty:
        print("No data to process after cleaning. Exiting pipeline run.")
        print("=== Pipeline Complete (No Data) ===\n")
        return

    # 3. Analyze sentiment
    df_analyzed = analyze_data(df_processed)
    
    # 4. Store in database
    db_columns = [
        'id', 'platform', 'title', 'text', 'author', 'created_utc', 
        'score', 'num_comments', 'url', 'sentiment_score', 'sentiment_label'
    ]
    
    df_to_insert = pd.DataFrame()
    for col in db_columns:
        if col in df_analyzed.columns:
            df_to_insert[col] = df_analyzed[col]
        else:
            df_to_insert[col] = None 

    db = SocialMediaDB()
    db.insert_posts(df_to_insert)
    
    print("=== Full Pipeline Complete ===\n")

if __name__ == "__main__":
    # Run immediately on start
    run_pipeline()
    
    # Schedule to run every 6 hours
    schedule.every(6).hours.do(run_pipeline)
    
    print("Pipeline scheduled. Running every 6 hours...")
    print("Press Ctrl+C to stop")
    
    while True:
        schedule.run_pending()
        time.sleep(60)
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

-----

## 📅 Day 7: Dashboard & Deployment

**Goal:** Create the interactive dashboard and all files needed for GitHub/deployment.

### File 1: `dashboard.py`

```bash
nano dashboard.py
```

Paste this code:

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from database import SocialMediaDB 
from datetime import date

def load_data():
    print("Loading data from database...")
    db = SocialMediaDB()
    df = db.get_all_posts()
    if df.empty:
        print("Database is empty. Dashboard will be empty.")
        return pd.DataFrame(columns=[
            'id', 'platform', 'created_utc', 'sentiment_label', 
            'sentiment_score', 'score', 'text'
        ])
        
    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
    df = df.dropna(subset=['created_utc']) # Drop rows where date conversion failed
    print(f"Loaded {len(df)} posts.")
    return df

df = load_data()

app = dash.Dash(__name__)

platform_options = [{'label': 'All Platforms', 'value': 'all'}]
if not df.empty:
    platform_options.extend([
        {'label': p.title(), 'value': p} for p in df['platform'].unique()
    ])

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1('Social Media Analytics Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Label('Select Platform:'),
            dcc.Dropdown(id='platform-dropdown', options=platform_options, value='all')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label('Select Date Range:'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['created_utc'].min().date() if not df.empty else date.today(),
                max_date_allowed=df['created_utc'].max().date() if not df.empty else date.today(),
                start_date=df['created_utc'].min().date() if not df.empty else date.today(),
                end_date=df['created_utc'].max().date() if not df.empty else date.today()
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'padding': '10px'})
    ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
    
    html.Div([
        dcc.Graph(id='sentiment-pie'),
        dcc.Graph(id='timeline'),
        dcc.Graph(id='sentiment-engagement')
    ])
])

@app.callback(
    [Output('sentiment-pie', 'figure'),
     Output('timeline', 'figure'),
     Output('sentiment-engagement', 'figure')],
    [Input('platform-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graphs(selected_platform, start_date, end_date):
    if df.empty: return {}, {}, {}
        
    filtered_df = df.copy()
    
    if selected_platform != 'all':
        filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['created_utc'].dt.date >= date.fromisoformat(start_date)) & 
            (filtered_df['created_utc'].dt.date <= date.fromisoformat(end_date))
        ]
    
    if filtered_df.empty: return {}, {}, {}

    pie_fig = px.pie(filtered_df, names='sentiment_label', title='Sentiment Distribution')
    
    timeline_df = filtered_df.set_index('created_utc').resample('D').size().reset_index(name='count')
    timeline_fig = px.bar(timeline_df, x='created_utc', y='count', title='Posts Over Time')
    
    scatter_fig = px.scatter(
        filtered_df.sample(n=min(len(filtered_df), 1000)), # Sample to avoid overplotting
        x='sentiment_score', 
        y='score', 
        color='platform',
        title='Sentiment vs Engagement (Sample of 1000)',
        labels={'score': 'Score (Likes/Upvotes)', 'sentiment_score': 'Sentiment'},
        hover_data=['text'] 
    )
    
    return pie_fig, timeline_fig, scatter_fig

if __name__ == '__main__':
    print("Dashboard running at http://127.0.0.1:8050")
    app.run_server(debug=True)
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 2: `requirements.txt`

*(This file is for deployment, listing all packages)*

```bash
nano requirements.txt
```

Paste this list (generated from `pip freeze`):

```
pandas
numpy
matplotlib
seaborn
praw
tweepy
google-api-python-client
textblob
vaderSentiment
SQLAlchemy
sqlite3
plotly
dash
schedule
python-dotenv
spacy
scikit-learn
nltk
joblib
gunicorn
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 3: `Procfile`

*(This file is for deployment, telling the server how to run the app)*

```bash
nano Procfile
```

Paste this **one line**:

```
web: gunicorn dashboard:app.server
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 4: `.gitignore`

*(This file tells Git to ignore secrets and data files)*

```bash
nano .gitignore
```

Paste this:

```
# Python
venv/
__pycache__/
*.pyc

# Secrets
.env

# Data & Models
*.csv
*.db
*.db-journal
*.zip
*.pkl

# OS files
.DS_Store
```

Save: `Ctrl+O`, Enter, `Ctrl+X`.

### File 5: `README.md`

*(This is the front page for your GitHub repository)*

```bash
nano README.md
```

Paste this template (and fill in your details):

````markdown
# 🚀 Social Media Analytics Pipeline

This is a complete ETL (Extract, Transform, Load) and data analysis pipeline built with Python. It collects data from Reddit, Twitter/X, and YouTube, processes and cleans the text, runs sentiment analysis, and displays the results on an interactive web dashboard.

## Features

* **Multi-Platform Collection:** Ingests data from Reddit (posts), Twitter/X (tweets), and YouTube (comments).
* **ETL Pipeline:** A scheduled, automated pipeline using `schedule` to run the collection process.
* **Data Storage:** Stores all processed data in a centralized **SQLite** database.
* **Advanced NLP:**
    * **Sentiment Analysis:** Uses VADER and a custom-trained `scikit-learn` model (Logistic Regression) to classify sentiment.
    * **Topic Modeling:** Uses LDA to discover hidden topics in the data.
    * **Entity Recognition:** Uses `spaCy` to extract key people, organizations, and places.
* **Interactive Dashboard:** A `Plotly Dash` dashboard that allows filtering by platform and date range.

## Tech Stack

* **Data Collection:** `praw` (Reddit), `tweepy` (Twitter), `google-api-python-client` (YouTube)
* **Data Processing:** `pandas`
* **NLP/ML:** `scikit-learn`, `spacy`, `nltk`, `vaderSentiment`, `joblib`
* **Database:** `sqlite3`
* **Dashboard:** `Plotly Dash`
* **Automation:** `schedule`
* **Deployment:** `gunicorn`

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_username/your_project_name.git](https://github.com/your_username/your_project_name.git)
    cd your_project_name
    ```
2.  **Create and activate environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLP models:**
    ```bash
    python3 -m spacy download en_core_web_sm
    python3 -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"
    ```
5.  **Create your `.env` file:**
    * Create a file named `.env` and add your API keys (see `Project Setup`).
6.  **(Optional) Train the custom model:**
    * Download the Sentiment140 dataset from Kaggle, name it `training_data.csv`.
    * Run `python3 train_model.py`.
7.  **Run the pipeline (to get data):**
    ```bash
    python3 pipeline.py
    ```
    *(Let it run once, then stop with Ctrl+C)*
8.  **Run the dashboard:**
    ```bash
    python3 dashboard.py
    ```
    * Visit `http://127.0.0.1:8050` in your browser.
````

Save: `Ctrl+O`, Enter, `Ctrl+X`.
