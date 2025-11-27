import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_data():
    """讀取主資料集與副資料集"""
    
    games_path = os.path.join(RAW_DATA_PATH, 'steam_games.csv')
    calendar_path = os.path.join(RAW_DATA_PATH, 'steam_calendar.csv')
    
    if not os.path.exists(games_path):
        raise FileNotFoundError(f"❌ can't find {games_path}")
    games_df = pd.read_csv(games_path)
    calendar_df = pd.read_csv(calendar_path)
    
    return games_df, calendar_df

def process_features(games_df, calendar_df):
    """特徵工程核心邏輯"""
    
    games_df['release_date'] = pd.to_datetime(games_df['release_date'], errors='coerce')
    games_df = games_df[games_df['release_date'] >= '2018-01-01'].copy() #可調整
    
    
    if 'positive_ratings' in games_df.columns:
        # def(可調整): 好評數 > 500 為熱門(2), > 50 為普通(1), 其餘冷門(0)
        conditions = [
            (games_df['positive_ratings'] >= 500),
            (games_df['positive_ratings'] >= 50)
        ]
        choices = [2, 1]
        games_df['success_level'] = np.select(conditions, choices, default=0)
    else:
        print("Can't find positive_ratings ")
    sale_dates = pd.to_datetime(calendar_df['date']).tolist()
    
    def check_is_sale(release_date):
        for sale_date in sale_dates:
            if abs((release_date - sale_date).days) <= 3:
                return 1
        return 0

    games_df['is_sale_period'] = games_df['release_date'].apply(check_is_sale)
    games_df['is_weekend'] = games_df['release_date'].dt.dayofweek.isin([5, 6]).astype(int) # 5=Sat, 6=Sun
    games_df['release_month'] = games_df['release_date'].dt.month

    # 'genres'
    target_tags = ['Action', 'Adventure', 'RPG', 'Strategy', 'Indie', 'Simulation']
    
    tag_col = 'genres' if 'genres' in games_df.columns else 'tags'
    
    if tag_col in games_df.columns:
        games_df[tag_col] = games_df[tag_col].fillna('')
        for tag in target_tags:
            games_df[f'tag_{tag}'] = games_df[tag_col].apply(lambda x: 1 if tag in str(x) else 0)

    numeric_cols = games_df.select_dtypes(include=[np.number]).columns.tolist()
    

    keep_cols = [c for c in numeric_cols if c not in ['appid']] 

    return games_df[keep_cols]

def main():
    try:
        df_games, df_calendar = load_data()
        df_clean = process_features(df_games, df_calendar)
        
        save_path = os.path.join(PROCESSED_DATA_PATH, 'training_data.csv')
        df_clean.to_csv(save_path, index=False)
        print(f"Save to: {save_path}")
        
    except Exception as e:
        print(f"Fail: {e}")

if __name__ == "__main__":
    main()