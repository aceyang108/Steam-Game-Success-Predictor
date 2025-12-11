import pandas as pd
import numpy as np
import os
from ast import literal_eval
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

INPUT_FILE = os.path.join(RAW_DATA_PATH, "games_march2025_cleaned.csv")

# -----------------------------------------------------
# 1. Load raw dataset
# -----------------------------------------------------
# Load the cleaned primary dataset containing game information.
# release_date is parsed as datetime for later feature extraction.
#
# 讀取主要的遊戲資料集（清洗後版本）。
# release_date 以 datetime 格式讀入，供後續特徵工程使用。
def load_dataset():
    df = pd.read_csv(INPUT_FILE, parse_dates=["release_date"])
    return df


# -----------------------------------------------------
# 2. Convert stringified lists into real Python lists
# -----------------------------------------------------
# Some fields (languages, developers, publishers, tags, genres, etc.)
# are stored as string representations of lists. Convert them into actual lists.
#
# 某些欄位（語言、開發商、發行商、標籤、類型等）為字串形式的 list，
# 本函式將其轉換成真正的 Python list。
def parse_list_columns(df, cols):
    for c in cols:
        df[c] = df[c].apply(lambda x: literal_eval(str(x)) if isinstance(x, str) else [])
    return df


# -----------------------------------------------------
# 3. Compute developer historical rating score (no leakage)
# -----------------------------------------------------

# developer_score = average user_rating of the same developer’s past games.
# Rules:
# - Only include games released before the current title (time-based).
# - Exclude the current game itself.
# - If a developer is working on multiple titles, explode then aggregate.
# - If no past data exists, fill with global mean.
#
# developer_score = 同一開發商「過去作品」的 user_rating 平均值。
# 規則：
# - 僅統計早於本遊戲發售的作品（時間上不會洩漏未來資料）。
# - 不包含本遊戲本身。
# - 一款遊戲可能有多位開發商，因此需先展開再聚合。
# - 若開發商沒有過去紀錄，則補上全體 user_rating 平均值。
def compute_developer_score(df):
    tmp = df[["appid", "release_date", "user_rating", "developers"]].explode("developers")
    tmp = tmp.sort_values(["developers", "release_date"])

    g = tmp.groupby("developers")["user_rating"]
    cum_sum = g.cumsum()
    cum_count = g.cumcount() + 1

    tmp["dev_score_past"] = (cum_sum - tmp["user_rating"]) / (cum_count - 1)

    global_mean = df["user_rating"].mean()
    tmp["dev_score_past"] = tmp["dev_score_past"].fillna(global_mean)

    dev_score_app = (
        tmp.groupby("appid")["dev_score_past"]
           .mean()
           .rename("developer_score")
    )

    df = df.merge(dev_score_app, on="appid", how="left")
    return df


# -----------------------------------------------------
# 4. Compute publisher historical features
# -----------------------------------------------------
# publisher_score = average past user_rating of the same publisher.
# publisher_game_count = number of previously released titles by the publisher.
# Similar rules as developer_score (time-based, no leakage).
#
# publisher_score = 同一發行商「過去作品」的 user_rating 平均值。
# publisher_game_count = 過去已發行作品的數量。
# 規則與 developer_score 相同（依時間排序、避免洩漏未來）。
def compute_publisher_features(df):
    tmp = df[["appid", "release_date", "user_rating", "publishers"]].explode("publishers")
    tmp = tmp.sort_values(["publishers", "release_date"])

    g = tmp.groupby("publishers")["user_rating"]

    cum_sum = g.cumsum()
    cum_count = g.cumcount() + 1

    tmp["pub_score_past"] = (cum_sum - tmp["user_rating"]) / (cum_count - 1)
    tmp["pub_game_count_past"] = g.cumcount()

    global_mean = df["user_rating"].mean()
    tmp["pub_score_past"] = tmp["pub_score_past"].fillna(global_mean)

    pub_agg = (
        tmp.groupby("appid")
           .agg(
               publisher_score=("pub_score_past", "mean"),
               publisher_game_count=("pub_game_count_past", "mean")
           )
    )

    df = df.merge(pub_agg, on="appid", how="left")
    return df

# -----------------------------------------------------
# 5. Merge wishlist features
# -----------------------------------------------------
# Merge wishlist-based features (rank, followers) into the main dataset by appid.
#
# 合併願望清單相關特徵（排名、追蹤人數）到主資料集（以 appid 對應）。
def add_wishlist_features(df, filename="wishlists_top1000.csv"):
    wishlist_path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(wishlist_path):
        print(f"Wishlist file not found, skip merge: {wishlist_path} / 找不到願望清單檔案，略過合併步驟")
        return df

    w = pd.read_csv(wishlist_path)

    # Ensure appid dtype is consistent before merge
    # 確保 appid 型別一致再進行合併
    if "appid" not in w.columns:
        print("Wishlist file has no 'appid' column, skip merge / 願望清單檔案缺少 'appid' 欄位，略過合併")
        return df

    # 盡量轉成跟主資料集相同的型別
    try:
        w["appid"] = w["appid"].astype(df["appid"].dtype)
    except Exception:
        # 如果強制轉型失敗，就先轉成字串再比對
        w["appid"] = w["appid"].astype(str)
        df["appid"] = df["appid"].astype(str)

    # Keep only columns we need
    # 只保留需要的欄位
    use_cols = ["appid"]
    if "rank" in w.columns:
        use_cols.append("rank")
    if "followers" in w.columns:
        use_cols.append("followers")

    w = w[use_cols]

    # Merge with left join, so that non-top1000 games remain in df with NaN wishlist values
    # 使用 left join，沒有出現在 top1000 的遊戲會保留，wishlist 欄位為 NaN
    df = df.merge(w, on="appid", how="left")

    # Rename to clearer feature names
    # 重新命名欄位為較清楚的特徵名稱
    rename_map = {}
    if "rank" in df.columns:
        rename_map["rank"] = "wishlist_rank"
    if "followers" in df.columns:
        rename_map["followers"] = "wishlist_followers"
    df = df.rename(columns=rename_map)

    # Handle missing wishlist values:
    # - For rank: games not in top1000 can be treated as having very low priority (large rank)
    # - For followers: missing means 0
    #
    # 處理缺失值：
    # - wishlist_rank：不在前 1000 名的遊戲，可視為排名很後面（給一個大的 rank）
    # - wishlist_followers：缺失視為 0
    if "wishlist_rank" in df.columns:
        max_rank = df["wishlist_rank"].max()
        df["wishlist_rank"] = df["wishlist_rank"].fillna(max_rank + 1)

    if "wishlist_followers" in df.columns:
        df["wishlist_followers"] = df["wishlist_followers"].fillna(0)

    return df

# -----------------------------------------------------
# 6. Build language-related features
# -----------------------------------------------------
# Includes:
# - Number of supported languages
# - Number of audio-supported languages
# - One-hot encode top-N most common languages (text and audio)
#
# 建立語言相關特徵，包括：
# - 支援語言數量
# - 支援語音語言數量
# - 最常見前 N 種語言的 one-hot 編碼（文字語言與語音語言）
def build_language_features(df, top_n=10):
    df["num_lang"] = df["supported_languages"].apply(len)
    df["num_audio_lang"] = df["full_audio_languages"].apply(len)

    lang_counter = Counter()
    df["supported_languages"].apply(lambda lst: lang_counter.update(lst))
    most_common_langs = [lang for lang, _ in lang_counter.most_common(top_n)]

    for lang in most_common_langs:
        df[f"lang_{lang}"] = df["supported_languages"].apply(lambda lst, l=lang: 1 if l in lst else 0)

    audio_counter = Counter()
    df["full_audio_languages"].apply(lambda lst: audio_counter.update(lst))
    most_common_audio = [lang for lang, _ in audio_counter.most_common(top_n)]

    for lang in most_common_audio:
        df[f"audio_lang_{lang}"] = df["full_audio_languages"].apply(lambda lst, l=lang: 1 if l in lst else 0)

    return df


# -----------------------------------------------------
# 7. One-hot encode genres / tags
# -----------------------------------------------------
# Convert multi-category fields (genres, tags) into multi-hot vectors.
#
# 將多類別（如 genres、tags）的欄位轉換成 multi-hot 向量。
def encode_multi_hot(df, col, prefix):
    all_values = set()
    df[col].apply(lambda lst: all_values.update(lst))
    all_values = sorted(all_values)

    for val in all_values:
        df[f"{prefix}_{val}"] = df[col].apply(lambda lst, v=val: 1 if v in lst else 0)
    return df


# -----------------------------------------------------
# 8. Date-related features
# -----------------------------------------------------
# Extract temporal features from release_date:
# - year, month, weekday, weekend flag
# - quarter, season
#
# 從發售日期萃取時間特徵：
# - 年、月、星期幾、是否週末
# - 季度、季節
def build_date_features(df):
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_dayofweek"] = df["release_date"].dt.dayofweek
    df["is_weekend"] = df["release_dayofweek"].isin([5, 6]).astype(int)
    df["release_quarter"] = df["release_date"].dt.quarter

    def month_to_season(m):
        if m in [3, 4, 5]:
            return 1
        elif m in [6, 7, 8]:
            return 2
        elif m in [9, 10, 11]:
            return 3
        return 4

    df["release_season"] = df["release_month"].apply(month_to_season)
    return df


# -----------------------------------------------------
# 9. Convert peak_ccu into classification labels (Y)
# -----------------------------------------------------
# success_level is defined from peak_ccu:
# - 0 = low popularity
# - 1 = medium
# - 2 = high
#
# 依照 peak_ccu 將遊戲分類為：
# - 0：冷門
# - 1：普通
# - 2：熱門
def compute_success_level(df, low=500, high=5000):
    if "peak_ccu" not in df.columns:
        raise ValueError("peak_ccu is missing; cannot compute success_level")

    def map_level(x):
        if x < low:
            return 0
        elif x < high:
            return 1
        return 2

    df["success_level"] = df["peak_ccu"].apply(map_level).astype(int)
    return df


# -----------------------------------------------------
# 10. Main preprocessing workflow
# -----------------------------------------------------
# Execute full preprocessing pipeline:
# - Load data
# - Prepare list-type columns
# - Build language, date, developer, and publisher features
# - Encode genres and tags
# - Compute success_level (classification target)
# - Remove extremely low-CCU games if needed
# - Assemble output feature columns and save to CSV
#
# 執行完整前處理流程：
# - 讀取資料
# - 處理 list 類型欄位
# - 建立語言、日期、開發商、發行商特徵
# - 編碼 genres 與 tags
# - 產生 success_level（模型用之 Y 值）
# - 若需要，可移除 peak_ccu 過低的遊戲
# - 組合最終特徵欄位並輸出 CSV
def preprocess():
    df = load_dataset()

    if "user_rating" not in df.columns:
        df["user_rating"] = df["positive"] / (df["positive"] + df["negative"]) * 100

    list_cols = [
        "supported_languages", "full_audio_languages",
        "developers", "publishers",
        "categories", "genres", "tags"
    ]
    df = parse_list_columns(df, list_cols)

    df["name_length"] = df["name"].astype(str).apply(len)

    df = build_language_features(df, top_n=10)
    df = build_date_features(df)
    df = compute_developer_score(df)
    df = compute_publisher_features(df)
    df = add_wishlist_features(df, filename="wishlists_top1000.csv")
    df = encode_multi_hot(df, "genres", "genre")
    df = encode_multi_hot(df, "tags", "tag")

    df = compute_success_level(df, low=500, high=5000)

    # Filter out extremely low-CCU games
    if "peak_ccu" in df.columns:
        before_len = len(df)
        df = df[df["peak_ccu"] >= 100].copy()
        after_len = len(df)
        print(f"Removed {before_len - after_len} rows with peak_ccu < 100")
    else:
        print("peak_ccu not found; skipping filtering step")

    if "popularity" in df.columns:
        df = df.drop(columns=["popularity"])

    BASE_FEATURES = [
        "appid",
        "price",
        "windows", "mac", "linux",
        "name_length",
        "num_lang", "num_audio_lang",
        "developer_score", "publisher_score", "publisher_game_count",
        "positive", "negative",
        "pct_pos_total", "num_reviews_total", "user_rating",
        "release_year", "release_month", "release_dayofweek",
        "is_weekend", "release_quarter", "release_season",
        "wishlist_rank", "wishlist_followers"
    ]

    GENRE_FEATURES = [c for c in df.columns if c.startswith("genre_")]
    TAG_FEATURES = [c for c in df.columns if c.startswith("tag_")]
    LANG_FEATURES = [c for c in df.columns if c.startswith("lang_") or c.startswith("audio_lang_")]

    OUTPUT_COLS = BASE_FEATURES + LANG_FEATURES + GENRE_FEATURES + TAG_FEATURES + ["success_level"]

    df_final = df[OUTPUT_COLS]

    save_path = os.path.join(PROCESSED_DATA_PATH, "data_after_preprocessing.csv")
    df_final.to_csv(save_path, index=False)

    print("Preprocessing completed")
    print(f"Output file saved to: {save_path}")
    print(f"Total rows: {len(df_final)}")
    print(f"Number of features (excluding success_level): {len(df_final.columns) - 1}")
    print("success_level distribution:")
    print(df_final["success_level"].value_counts().sort_index())


if __name__ == "__main__":
    preprocess()