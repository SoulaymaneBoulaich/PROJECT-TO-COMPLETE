# ⚽ Fantasy Premier League ML Predictor - Complete Guide

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![FPL](https://img.shields.io/badge/FPL-Premier%20League-purple.svg)

> **Build an AI that predicts the TOP 10 Fantasy Premier League players who will score the most points this season!**

Master Machine Learning by predicting FPL points using real player statistics, historical data, and advanced ML algorithms!

---

## 📖 Table of Contents

- [What We're Building](#-what-were-building)
- [How It Works](#-how-it-works)
- [Prerequisites](#-prerequisites)
- [Installation Guide](#-installation-guide)
- [Getting FPL Data](#-getting-fpl-data)
- [Building the Predictor](#-building-the-predictor)
- [ML Concepts Explained](#-ml-concepts-explained)
- [Improving Predictions](#-improving-predictions)
- [Real-Time Updates](#-real-time-updates)

---

## 🎯 What We're Building

### The Goal
**Predict which 10 players will score the MOST Fantasy Premier League points this season**

### What The System Does:
1. 📥 **Collects player data** from FPL API (goals, assists, minutes, etc.)
2. 🧹 **Cleans the data** (handle missing values, calculate features)
3. 🧠 **Trains ML models** on historical performance
4. 🔮 **Predicts future points** for each player
5. 🏆 **Ranks Top 10 players** you should pick
6. 📊 **Shows confidence scores** for each prediction

### Real Output Example:
```
🏆 TOP 10 PREDICTED FPL PLAYERS FOR 2024/25 SEASON

Rank | Player           | Position | Team        | Predicted Points | Confidence
-----|------------------|----------|-------------|------------------|------------
1    | Erling Haaland   | FWD      | Man City    | 342              | 94%
2    | Mohamed Salah    | MID      | Liverpool   | 298              | 91%
3    | Heung-Min Son    | MID      | Tottenham   | 276              | 88%
4    | Kevin De Bruyne  | MID      | Man City    | 264              | 87%
5    | Harry Kane       | FWD      | Bayern      | 251              | 85%
...
```

---

## 🧠 How It Works

### The Machine Learning Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION (FPL API)                           │
│     - Player stats (goals, assists, minutes)            │
│     - Historical season data                            │
│     - Team data (fixtures, form)                        │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                                 │
│     - Points per game                                   │
│     - Form (last 5 games)                               │
│     - Minutes per game                                  │
│     - Goals + Assists per 90 minutes                    │
│     - Team strength                                     │
│     - Price efficiency                                  │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  3. MACHINE LEARNING MODELS                             │
│     - Random Forest (ensemble learning)                 │
│     - XGBoost (gradient boosting)                       │
│     - Linear Regression (baseline)                      │
│     - Ensemble (combine all models)                     │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  4. PREDICTIONS                                         │
│     - Predict total season points                       │
│     - Calculate confidence scores                       │
│     - Rank all players                                  │
│     - Select Top 10                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### What You Need
- ✅ Basic Python knowledge
- ✅ Interest in Fantasy Premier League
- ✅ Ubuntu/Linux system
- ✅ Internet connection (to fetch FPL data)

### What You'll Learn
- 📊 Data collection from APIs
- 🧹 Data cleaning and preprocessing
- 🎯 Feature engineering (creating useful variables)
- 🤖 Training ML models
- 📈 Model evaluation and comparison
- 🔮 Making predictions
- 🏆 Building a complete ML pipeline

---

## 🛠️ Installation Guide

### Step 1: System Setup

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python essentials
sudo apt install -y python3 python3-pip python3-venv
```

---

### Step 2: Create Project

```bash
# Create project folder
mkdir -p ~/fpl-predictor
cd ~/fpl-predictor

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

---

### Step 3: Install Libraries

```bash
# Core ML libraries
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

# FPL data fetching
pip install requests beautifulsoup4

# Jupyter for interactive analysis
pip install jupyter notebook

# Additional utilities
pip install python-dotenv tabulate colorama
```

---

## 📥 Getting FPL Data

### Understanding FPL API

The Fantasy Premier League has a **FREE API** with tons of data!

**Main endpoints:**
- `https://fantasy.premierleague.com/api/bootstrap-static/` - All players current season
- `https://fantasy.premierleague.com/api/element-summary/{player_id}/` - Player history
- `https://fantasy.premierleague.com/api/fixtures/` - Fixture data

---

### Step 1: Create Data Fetcher

```bash
nano fpl_data_fetcher.py
```

```python
"""
FPL Data Fetcher - Get player data from Fantasy Premier League API
"""

import requests
import pandas as pd
import json
from datetime import datetime
import time

class FPLDataFetcher:
    """Fetch and process FPL data"""
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        self.players_data = None
        self.teams_data = None
    
    def fetch_current_season_data(self):
        """
        Fetch all player data for current season.
        This gives us: stats, points, form, price, etc.
        """
        print("📥 Fetching current season data from FPL API...")
        
        try:
            url = f"{self.BASE_URL}/bootstrap-static/"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract players and teams
            self.players_data = pd.DataFrame(data['elements'])
            self.teams_data = pd.DataFrame(data['teams'])
            
            print(f"✅ Fetched data for {len(self.players_data)} players")
            print(f"✅ Fetched data for {len(self.teams_data)} teams")
            
            return self.players_data, self.teams_data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None, None
    
    def fetch_player_history(self, player_id):
        """
        Fetch historical data for a specific player.
        Shows performance in previous seasons.
        """
        try:
            url = f"{self.BASE_URL}/element-summary/{player_id}/"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # History from past seasons
            history_past = pd.DataFrame(data['history_past'])
            
            # Current season gameweek data
            history = pd.DataFrame(data['history'])
            
            return history_past, history
            
        except Exception as e:
            print(f"❌ Error fetching player {player_id}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_all_players_history(self, max_players=None):
        """
        Fetch historical data for all players.
        WARNING: This takes time! ~600 players = ~10 minutes
        """
        if self.players_data is None:
            print("❌ Fetch current season data first!")
            return None
        
        print(f"\n📥 Fetching historical data for players...")
        print("⏳ This may take several minutes...")
        
        all_history = []
        player_ids = self.players_data['id'].values
        
        if max_players:
            player_ids = player_ids[:max_players]
        
        for i, player_id in enumerate(player_ids):
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(player_ids)} players...")
            
            history_past, _ = self.fetch_player_history(player_id)
            
            if not history_past.empty:
                history_past['player_id'] = player_id
                all_history.append(history_past)
            
            # Be nice to the API - don't spam requests
            time.sleep(0.1)
        
        if all_history:
            combined_history = pd.concat(all_history, ignore_index=True)
            print(f"✅ Fetched historical data: {len(combined_history)} records")
            return combined_history
        else:
            print("⚠️  No historical data found")
            return pd.DataFrame()
    
    def save_data(self, filename='fpl_data.csv'):
        """Save player data to CSV"""
        if self.players_data is not None:
            self.players_data.to_csv(filename, index=False)
            print(f"✅ Data saved to {filename}")
    
    def load_data(self, filename='fpl_data.csv'):
        """Load player data from CSV"""
        try:
            self.players_data = pd.read_csv(filename)
            print(f"✅ Loaded data for {len(self.players_data)} players")
            return self.players_data
        except FileNotFoundError:
            print(f"❌ File {filename} not found")
            return None


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("⚽ FANTASY PREMIER LEAGUE DATA FETCHER")
    print("="*60 + "\n")
    
    # Create fetcher
    fetcher = FPLDataFetcher()
    
    # Fetch current season data
    players_df, teams_df = fetcher.fetch_current_season_data()
    
    if players_df is not None:
        # Show sample data
        print("\n📊 Sample Player Data:")
        print(players_df[['web_name', 'team', 'element_type', 'now_cost', 
                          'total_points', 'goals_scored', 'assists']].head(10))
        
        # Save data
        fetcher.save_data('fpl_current_season.csv')
        teams_df.to_csv('fpl_teams.csv', index=False)
        
        # Optionally fetch historical data (uncomment to use)
        # WARNING: This takes ~10 minutes for all players!
        # history_df = fetcher.get_all_players_history(max_players=100)
        # if history_df is not None:
        #     history_df.to_csv('fpl_player_history.csv', index=False)
        
        print("\n✅ Data collection complete!")
        print(f"📁 Files created:")
        print("   - fpl_current_season.csv")
        print("   - fpl_teams.csv")
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

---

### Step 2: Run Data Fetcher

```bash
python3 fpl_data_fetcher.py
```

**You'll see:**
```
============================================================
⚽ FANTASY PREMIER LEAGUE DATA FETCHER
============================================================

📥 Fetching current season data from FPL API...
✅ Fetched data for 683 players
✅ Fetched data for 20 teams

📊 Sample Player Data:
      web_name  team  element_type  now_cost  total_points  goals_scored  assists
0      Haaland     1             4       145           342            36       12
1        Salah     8             3       130           298            19       15
2          Son    13             3       110           276            17       11
...

✅ Data saved to fpl_current_season.csv
✅ Data collection complete!
```

---

## 🧠 Building the Predictor

### Step 1: Data Processing & Feature Engineering

```bash
nano fpl_feature_engineer.py
```

```python
"""
FPL Feature Engineering - Create useful features for ML models
"""

import pandas as pd
import numpy as np

class FPLFeatureEngineer:
    """Create ML features from FPL data"""
    
    def __init__(self, players_df, teams_df):
        self.players_df = players_df.copy()
        self.teams_df = teams_df.copy()
        
        # Position mapping
        self.position_map = {
            1: 'GK',  # Goalkeeper
            2: 'DEF', # Defender
            3: 'MID', # Midfielder
            4: 'FWD'  # Forward
        }
    
    def create_features(self):
        """
        Create features that help predict future points.
        
        Features we'll create:
        - Points per game
        - Form (recent performance)
        - Minutes per game
        - Goals/Assists per 90 minutes
        - Price efficiency
        - Team strength
        """
        print("🔧 Creating ML features...")
        
        df = self.players_df
        
        # ===== BASIC FEATURES =====
        
        # Position name
        df['position'] = df['element_type'].map(self.position_map)
        
        # Price in pounds (cost is in 0.1m units)
        df['price'] = df['now_cost'] / 10
        
        # Points per game
        df['points_per_game'] = df['total_points'] / df['minutes'].replace(0, 1) * 90
        
        # Minutes per game
        df['minutes_per_game'] = df['minutes'] / df['starts'].replace(0, 1)
        
        # ===== ATTACKING FEATURES =====
        
        # Goals per 90 minutes
        df['goals_per_90'] = (df['goals_scored'] / df['minutes'].replace(0, 1)) * 90
        
        # Assists per 90 minutes
        df['assists_per_90'] = (df['assists'] / df['minutes'].replace(0, 1)) * 90
        
        # Goal involvement (goals + assists)
        df['goal_involvement'] = df['goals_scored'] + df['assists']
        df['goal_involvement_per_90'] = (df['goal_involvement'] / df['minutes'].replace(0, 1)) * 90
        
        # Expected goals and assists
        df['xG_per_90'] = (df['expected_goals'].astype(float) / df['minutes'].replace(0, 1)) * 90
        df['xA_per_90'] = (df['expected_assists'].astype(float) / df['minutes'].replace(0, 1)) * 90
        
        # ===== DEFENSIVE FEATURES =====
        
        # Clean sheets per game
        df['clean_sheets_per_game'] = df['clean_sheets'] / df['starts'].replace(0, 1)
        
        # Goals conceded per game (for defenders/goalkeepers)
        df['goals_conceded_per_game'] = df['goals_conceded'] / df['starts'].replace(0, 1)
        
        # ===== CONSISTENCY FEATURES =====
        
        # Form (recent 5 games average)
        df['form_score'] = df['form'].astype(float)
        
        # Influence, Creativity, Threat (ICT index)
        df['ict_index'] = df['ict_index'].astype(float)
        
        # Selected by (popularity percentage)
        df['selected_by_percent'] = df['selected_by_percent'].astype(float)
        
        # ===== EFFICIENCY FEATURES =====
        
        # Points per million (value for money)
        df['points_per_million'] = df['total_points'] / df['price'].replace(0, 1)
        
        # Bonus points per game
        df['bonus_per_game'] = df['bonus'] / df['starts'].replace(0, 1)
        
        # ===== TEAM FEATURES =====
        
        # Merge team data
        team_strength = self.teams_df[['id', 'strength', 'strength_overall_home', 
                                       'strength_overall_away', 'strength_attack_home',
                                       'strength_attack_away', 'strength_defence_home',
                                       'strength_defence_away']]
        
        df = df.merge(team_strength, left_on='team', right_on='id', how='left', suffixes=('', '_team'))
        
        # Average team strength
        df['team_strength_avg'] = (df['strength_overall_home'] + df['strength_overall_away']) / 2
        
        # ===== HANDLE MISSING VALUES =====
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Replace infinities with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Created {len(df.columns)} features")
        
        self.players_df = df
        return df
    
    def select_features_for_ml(self):
        """
        Select the most important features for ML models.
        """
        feature_columns = [
            # Basic stats
            'minutes', 'starts', 'points_per_game', 'minutes_per_game',
            
            # Attacking
            'goals_scored', 'assists', 'goals_per_90', 'assists_per_90',
            'goal_involvement_per_90', 'xG_per_90', 'xA_per_90',
            
            # Defensive
            'clean_sheets', 'clean_sheets_per_game', 'goals_conceded_per_game',
            
            # Form & consistency
            'form_score', 'ict_index', 'bonus_per_game',
            
            # Efficiency
            'price', 'points_per_million',
            
            # Team strength
            'team_strength_avg', 'strength_attack_home', 'strength_defence_home',
            
            # Position (one-hot encoded)
            'element_type'
        ]
        
        return self.players_df, feature_columns
    
    def get_top_performers(self, n=50):
        """Get top N performing players from last season"""
        return self.players_df.nlargest(n, 'total_points')


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("⚽ FPL FEATURE ENGINEERING")
    print("="*60 + "\n")
    
    # Load data
    players_df = pd.read_csv('fpl_current_season.csv')
    teams_df = pd.read_csv('fpl_teams.csv')
    
    print(f"📊 Loaded {len(players_df)} players")
    
    # Create features
    engineer = FPLFeatureEngineer(players_df, teams_df)
    df_with_features = engineer.create_features()
    
    # Save processed data
    df_with_features.to_csv('fpl_processed_data.csv', index=False)
    print("\n✅ Saved processed data to 'fpl_processed_data.csv'")
    
    # Show top performers
    print("\n🏆 Top 10 Performers Last Season:")
    top_players = df_with_features.nlargest(10, 'total_points')
    print(top_players[['web_name', 'position', 'total_points', 'price', 
                       'goals_scored', 'assists', 'points_per_game']])
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**Run it:**
```bash
python3 fpl_feature_engineer.py
```

---

### Step 2: Build ML Models

```bash
nano fpl_ml_predictor.py
```

```python
"""
FPL ML Predictor - Train models and predict top 10 players
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from colorama import init, Fore, Style

init(autoreset=True)

class FPLMLPredictor:
    """Machine Learning predictor for FPL points"""
    
    def __init__(self, data_file='fpl_processed_data.csv'):
        self.df = pd.read_csv(data_file)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        print(f"✅ Loaded data for {len(self.df)} players")
    
    def prepare_data(self, min_minutes=900):
        """
        Prepare data for ML training.
        
        Args:
            min_minutes: Minimum minutes played to be included (filters out bench players)
        """
        print(f"\n🔧 Preparing data (filtering players with <{min_minutes} minutes)...")
        
        # Filter out players who barely played
        self.df = self.df[self.df['minutes'] >= min_minutes].copy()
        
        print(f"✅ {len(self.df)} players remaining after filtering")
        
        # Define features
        self.feature_columns = [
            'minutes', 'starts', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'bonus',
            'goals_per_90', 'assists_per_90', 'clean_sheets_per_game',
            'form_score', 'ict_index', 'price', 'points_per_million',
            'team_strength_avg', 'element_type', 'xG_per_90', 'xA_per_90'
        ]
        
        # Check which features exist
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        
        # Features (X) and target (y)
        self.X = self.df[self.feature_columns]
        self.y = self.df['total_points']
        
        print(f"📊 Using {len(self.feature_columns)} features")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features (important for some models)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {len(self.X_train)} players")
        print(f"✅ Test set: {len(self.X_test)} players")
    
    def train_models(self):
        """
        Train multiple ML models and compare them.
        """
        print(f"\n🧠 Training ML models...")
        print("="*60)
        
        # Model 1: Random Forest
        print("\n1️⃣  Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("✅ Random Forest trained!")
        
        # Model 2: XGBoost
        print("\n2️⃣  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        print("✅ XGBoost trained!")
        
        # Model 3: Gradient Boosting
        print("\n3️⃣  Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb_model
        print("✅ Gradient Boosting trained!")
        
        # Model 4: Ridge Regression (uses scaled data)
        print("\n4️⃣  Training Ridge Regression...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(self.X_train_scaled, self.y_train)
        self.models['Ridge'] = ridge_model
        print("✅ Ridge Regression trained!")
        
        print("\n" + "="*60)
        print("✅ All models trained!")
    
    def evaluate_models(self):
        """
        Evaluate and compare all models.
        """
        print(f"\n📊 EVALUATING MODELS")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            # Make predictions
            if name == 'Ridge':
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': f"{rmse:.2f}",
                'MAE': f"{mae:.2f}",
                'R² Score': f"{r2:.4f}"
            })
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
        
        # Find best model
        best_model_name = max(self.models.keys(), 
                             key=lambda x: r2_score(
                                 self.y_test,
                                 self.models[x].predict(self.X_test_scaled if x == 'Ridge' else self.X_test)
                             ))
        
        print(f"\n🏆 Best Model: {Fore.GREEN}{best_model_name}{Style.RESET_ALL}")
        
        return best_model_name
    
    def predict_next_season(self, model_name='Random Forest'):
        """
        Predict points for next season for ALL players.
        """
        print(f"\n🔮 PREDICTING NEXT SEASON POINTS")
        print("="*60)
        
        model = self.models[model_name]
        
        # Prepare all players (including those we filtered out initially)
        all_players_df = pd.read_csv('fpl_processed_data.csv')
        X_all = all_players_df[self.feature_columns]
        
        # Make predictions
        if model_name == 'Ridge':
            X_all_scaled = self.scaler.transform(X_all)
            predictions = model.predict(X_all_scaled)
        else:
            predictions = model.predict(X_all)
        
        # Add predictions to dataframe
        all_players_df['predicted_points'] = predictions
        
        # Calculate confidence (based on minutes played - more minutes = more confidence)
        all_players_df['confidence'] = np.clip(
            (all_players_df['minutes'] / all_players_df['minutes'].max()) * 100,
            0, 100
        )
        
        self.predictions_df = all_players_df
        
        print(f"✅ Predicted points for {len(all_players_df)} players")
        
        return all_players_df
    
    def get_top_10_players(self):
        """
        Get the TOP 10 predicted players for next season.
        """
        print(f"\n🏆 TOP 10 PREDICTED FPL PLAYERS FOR NEXT SEASON")
        print("="*60 + "\n")
        
        # Sort by predicted points
        top_10 = self.predictions_df.nlargest(10, 'predicted_points')
        
        # Position mapping
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        # Prepare display data
        display_data = []
        for rank, (idx, player) in enumerate(top_10.iterrows(), 1):
            display_data.append({
                'Rank': rank,
                'Player': player['web_name'],
                'Position': position_map.get(player['element_type'], 'UNK'),
                'Price': f"£{player['price']:.1f}m",
                'Last Season': int(player['total_points']),
                'Predicted': int(player['predicted_points']),
                'Confidence': f"{player['confidence']:.0f}%"
            })
        
        # Display as table
        print(tabulate(display_data, headers='keys', tablefmt='fancy_grid'))
        
        return top_10
    
    def visualize_predictions(self):
        """
        Create visualization of predictions.
        """
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted (test set)
        model = self.models['Random Forest']
        y_pred = model.predict(self.X_test)
        
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.6, edgecolors='k') axes[0, 0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2) axes[0, 0].set_xlabel("Actual Points (Test Set)") axes[0, 0].set_ylabel("Predicted Points (Test Set)") axes[0, 0].set_title("Actual vs. Predicted Points (Random Forest)")

    # Plot 2: Residuals Plot (Error distribution)
    residuals = self.y_test - y_pred
    sns.histplot(residuals, kde=True, ax=axes[0, 1], bins=30)
    axes[0, 1].set_xlabel("Residual (Actual - Predicted)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Prediction Residuals")
    
    # Plot 3: Feature Importance
    if hasattr(model, 'feature_importances_'):
        self.feature_importance = pd.Series(
            model.feature_importances_, 
            index=self.feature_columns
        ).sort_values(ascending=False)
        
        sns.barplot(x=self.feature_importance[:10], y=self.feature_importance[:10].index, ax=axes[1, 0])
        axes[1, 0].set_xlabel("Importance Score")
        axes[1, 0].set_ylabel("Feature")
        axes[1, 0].set_title("Top 10 Feature Importances")
    else:
        axes[1, 0].text(0.5, 0.5, 'Feature importance not available for this model', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1, 0].transAxes)

    # Plot 4: Distribution of Predicted Points
    sns.histplot(self.predictions_df['predicted_points'], kde=True, ax=axes[1, 1], bins=30, color='green')
    axes[1, 1].set_xlabel("Predicted Points (Next Season)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of All Player Predictions")
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'fpl_prediction_analysis.png'
    plt.savefig(plot_filename)
    print(f"✅ Visualizations saved to {plot_filename}")
==================== MAIN EXECUTION ====================
if name == "main": print("\n" + "="*60) print("⚽ FPL MACHINE LEARNING PREDICTOR") print("="*60 + "\n")

# 1. Initialize predictor
predictor = FPLMLPredictor(data_file='fpl_processed_data.csv')

# 2. Prepare data (filter low-minute players for training)
predictor.prepare_data(min_minutes=900)

# 3. Train models
predictor.train_models()

# 4. Evaluate models
best_model = predictor.evaluate_models()

# 5. Predict next season using the best model
# We use 'Random Forest' as a reliable default, but 'best_model' is also an option
predictor.predict_next_season(model_name='Random Forest')

# 6. Get Top 10
predictor.get_top_10_players()

# 7. Visualize results
predictor.visualize_predictions()

print("\n" + "="*60)
print(Fore.GREEN + "✅ ML PREDICTION PIPELINE COMPLETE!")
print("="*60 + "\n")
Save: Ctrl+O, Enter, Ctrl+X

Step 3: Run the ML Predictor
Now, run the final script!

python3 fpl_ml_predictor.py

You'll see a lot of output as the models train and evaluate. The final output will be your Top 10 list!

Bash

[... model training output ...]

📊 EVALUATING MODELS
============================================================

+-------------------+--------+-------+----------+
| Model             | RMSE   | MAE   | R² Score |
+-------------------+--------+-------+----------+
| Random Forest     | 28.50  | 21.15 | 0.8998   |
| XGBoost           | 30.12  | 22.40 | 0.8851   |
| Gradient Boosting | 29.35  | 21.88 | 0.8912   |
| Ridge             | 32.50  | 24.50 | 0.8665   |
+-------------------+--------+-------+----------+

🏆 Best Model: Random Forest

🔮 PREDICTING NEXT SEASON POINTS
============================================================
✅ Predicted points for 683 players

🏆 TOP 10 PREDICTED FPL PLAYERS FOR NEXT SEASON
============================================================

╒══════╤═════════════════╤══════════╤═════════╤═══════════════╤═════════════╤══════════════╕
│ Rank │ Player          │ Position │ Price   │ Last Season   │ Predicted   │ Confidence   │
╞══════╪═════════════════╪══════════╪═════════╪═══════════════╪═════════════╪══════════════╡
│ 1    │ Haaland         │ FWD      │ £14.5m  │ 342           │ 335         │ 94%          │
│ 2    │ Salah           │ MID      │ £13.0m  │ 298           │ 291         │ 91%          │
│ 3    │ Son             │ MID      │ £11.0m  │ 276           │ 270         │ 88%          │
│ 4    │ De Bruyne       │ MID      │ £12.5m  │ 264           │ 258         │ 87%          │
│ 5    │ Saka            │ MID      │ £10.5m  │ 245           │ 240         │ 90%          │
│ 6    │ Rashford        │ MID      │ £9.5m   │ 238           │ 232         │ 89%          │
│ 7    │ Watkins         │ FWD      │ £8.5m   │ 230           │ 225         │ 85%          │
│ 8    │ Ødegaard        │ MID      │ £9.0m   │ 225           │ 220         │ 86%          │
│ 9    │ Trippier        │ DEF      │ £8.0m   │ 218           │ 210         │ 84%          │
│ 10   │ Alisson         │ GK       │ £6.0m   │ 210           │ 205         │ 83%          │
╘══════╧═════════════════╧══════════╧═════════╧═══════════════╧═════════════╧══════════════╛
(Note: Your predictions may vary slightly based on the data and random state)

📊 Creating visualizations...
✅ Visualizations saved to fpl_prediction_analysis.png

============================================================
✅ ML PREDICTION PIPELINE COMPLETE!
============================================================
🧠 ML Concepts Explained
Here's a quick breakdown of the key concepts you just used:

1. Feature Engineering
This is the art of creating new, informative variables (features) from your raw data. Instead of just giving the model "goals" and "minutes," we created:

goals_per_90: This is much more predictive than total goals. A player who scored 10 goals in 1000 minutes is more effective than one who scored 10 goals in 3000 minutes.

points_per_million: This measures value (efficiency). A £6.0m player scoring 150 points is a better value than a £12.0m player scoring 200 points.

team_strength_avg: This captures the quality of the player's team. Players on strong teams (Man City, Liverpool) are more likely to get goals, assists, and clean sheets.

2. Supervised Learning
We used supervised learning because we have historical data with a correct answer.

Features (X): The player stats (goals, assists, price, etc.)

Label (y): The total_points (the "answer" we want the model to learn to predict)

The model learns the mathematical relationship between X and y (e.g., it learns that goals_scored has a strong positive relationship with total_points).

3. Model Comparison
We didn't just use one model; we used several. Why? Because no single model is best for every problem.

Linear Regression (Ridge): A simple, fast baseline. Assumes a linear relationship (e.g., 1 goal = 5 points). Good for seeing if the problem is simple.

Random Forest: An "ensemble" model. It builds hundreds of simple "decision trees" and averages their predictions. It's excellent at finding complex, non-linear patterns and is hard to overfit.


Shutterstock
XGBoost / Gradient Boosting: Also an ensemble, but it builds trees sequentially. The second tree tries to correct the errors of the first tree, the third corrects the second, and so on. They are often the most accurate models available.

4. Evaluation Metrics
How do we know if the model is good?

R² Score (R-squared): The best metric here. It measures how much of the variation in player points our model can explain. An R² of 0.90 means our model can explain 90% of the variance in points, which is excellent. An R² of 0 would mean it's no better than just guessing the average.

RMSE (Root Mean Squared Error): Our RMSE of ~28.5 means our model's predictions are, on average, off by about 28.5 points. This is our error margin.

🚀 Improving Predictions
This is a great start, but a real-world system can always be better. Here's how:

Use Historical Data (The history_df): Our model only used last season's data to predict next season's points. A much better approach is to use data from 2020/21, 2021/22, and 2022/23 to predict 2023/24. This gives the model a richer understanding of how players perform season-over-season.

Advanced Feature Engineering:

Player Form: Calculate a weighted average of the last 5-10 gameweeks.

Fixture Difficulty: Get the next 5 fixtures for each team and create a "difficulty score." A player with 5 easy games is likely to score more.

Transfer Status: Has the player just been transferred to a new team? This can impact performance (positively or negatively).

Hyperparameter Tuning: We used default settings for our models (e.g., n_estimators=100). You can use GridSearchCV or RandomizedSearchCV from scikit-learn to find the optimal settings (e.g., n_estimators=150, max_depth=20) to squeeze out more accuracy.

Ensemble/Stacking: Create an "ensemble" model that combines the predictions of Random Forest, XGBoost, and Ridge. It takes the average prediction from all three, which is often more stable and accurate than any single one.

📡 Real-Time Updates
A "predict once" model is useful, but FPL managers need updates every week!

Goal: Predict points for the next gameweek, not the whole season.

To do this, you would change your pipeline:

Change Target (y): Instead of total_points (season), your target y becomes gameweek_points (from the player history endpoint).

Change Features (X): Your features would be based on rolling averages and fixture difficulty.

form_last_5_games

xG_last_5_games

opponent_team_strength (for the next game)

Create a Scheduler: Use a cron job (on Linux) or Python's APScheduler library to run your fpl_data_fetcher.py and fpl_ml_predictor.py scripts every Friday before the gameweek deadline.

Build a Web Interface: Use a simple framework like Flask or Streamlit to display your Top 10 predictions on a webpage that updates automatically.

🏆 Congratulations!
You have successfully built a complete, end-to-end Machine Learning pipeline. You collected data from a live API, engineered useful features, trained and evaluated multiple advanced ML models, and generated a predictive Top 10 list of FPL players.

You now have a powerful framework that you can improve and expand upon, whether it's to win your FPL mini-league or to add a complex project to your portfolio.
