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

#output file path
OUTPUT_FILE_MAIN = os.path.join(PROCESSED_DATA_PATH, "training_data_main.csv") # 僅 Kaggle
OUTPUT_FILE_FULL = os.path.join(PROCESSED_DATA_PATH, "training_data_full.csv") # Kaggle + SteamDB
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
# 3. Compute developer historical features (no leakage)
# -----------------------------------------------------
# Build developer history features using only games released BEFORE the current game.
# This prevents target leakage by not using "future" information.
#
# Features (per game):
# - developer_score: average past user_rating of the same developer(s)
# - developer_game_count: number of past released games by the same developer(s)
# - developer_avg_reviews: average past num_reviews_total of the same developer(s)
# - developer_avg_recommendations: average past recommendations of the same developer(s)
#
# Use explode() because one game may have multiple developers, then aggregate back to appid.
# If a developer has no past games, fill with global mean (for averages) and 0 for counts.
#
# 僅使用「發售日期早於本作」的同開發商作品來計算履歷特徵，避免洩漏未來資訊。
#
# 每款遊戲輸出的特徵包含：
# - developer_score：同開發商過去作品平均 user_rating
# - developer_game_count：同開發商過去作品數
# - developer_avg_reviews：同開發商過去作品平均 num_reviews_total
# - developer_avg_recommendations：同開發商過去作品平均 recommendations
#
# 因一款遊戲可能有多位開發商，先 explode() 展開再聚合回 appid。
# 若開發商沒有過去作品：平均值用全局平均補，作品數用 0。
def compute_developer_score(df):
    required_cols = ["appid", "release_date", "developers", "user_rating", "num_reviews_total", "recommendations"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for developer history features: {missing}")

    tmp = df[required_cols].explode("developers")
    tmp = tmp.sort_values(["developers", "release_date"])

    g = tmp.groupby("developers", sort=False)

    # Count of past games for each developer at each row (excluding current game)
    # 每一列代表該開發商在「本作之前」已發行幾款作品（不含本作）
    tmp["dev_game_count_past"] = g.cumcount()

    # Past sums (exclude current row)
    # 過去累積和（扣掉本作，避免洩漏）
    tmp["dev_rating_sum_past"] = g["user_rating"].cumsum() - tmp["user_rating"]
    tmp["dev_reviews_sum_past"] = g["num_reviews_total"].cumsum() - tmp["num_reviews_total"]
    tmp["dev_reco_sum_past"] = g["recommendations"].cumsum() - tmp["recommendations"]

    # Past averages; division by 0 will yield NaN for first game of each developer
    # 過去平均；第一款作品會除以 0 → NaN，後續會用全局平均補
    tmp["developer_score"] = tmp["dev_rating_sum_past"] / tmp["dev_game_count_past"]
    tmp["developer_avg_reviews"] = tmp["dev_reviews_sum_past"] / tmp["dev_game_count_past"]
    tmp["developer_avg_recommendations"] = tmp["dev_reco_sum_past"] / tmp["dev_game_count_past"]

    global_rating = df["user_rating"].mean()
    global_reviews = df["num_reviews_total"].mean()
    global_reco = df["recommendations"].mean()

    tmp["developer_score"] = tmp["developer_score"].fillna(global_rating)
    tmp["developer_avg_reviews"] = tmp["developer_avg_reviews"].fillna(global_reviews)
    tmp["developer_avg_recommendations"] = tmp["developer_avg_recommendations"].fillna(global_reco)

    # Aggregate multiple developers back to appid (average across developers)
    # 多開發商時，對同一 appid 取平均
    dev_agg = (
        tmp.groupby("appid")
           .agg(
               developer_score=("developer_score", "mean"),
               developer_game_count=("dev_game_count_past", "mean"),
               developer_avg_reviews=("developer_avg_reviews", "mean"),
               developer_avg_recommendations=("developer_avg_recommendations", "mean"),
           )
    )

    df = df.merge(dev_agg, on="appid", how="left")
    return df


# -----------------------------------------------------
# 4. Compute publisher historical features (no leakage)
# -----------------------------------------------------
# Build publisher history features using only games released BEFORE the current game.
#
# Features (per game):
# - publisher_score: average past user_rating of the same publisher(s)
# - publisher_game_count: number of past released games by the same publisher(s)
# - publisher_avg_reviews: average past num_reviews_total of the same publisher(s)
#
# Use explode() because one game may have multiple publishers, then aggregate back to appid.
# If a publisher has no past games, fill with global mean (for averages) and 0 for counts.
#
# 僅使用「發售日期早於本作」的同發行商作品來計算履歷特徵，避免洩漏未來資訊。
#
# 每款遊戲輸出的特徵包含：
# - publisher_score：同發行商過去作品平均 user_rating
# - publisher_game_count：同發行商過去作品數
# - publisher_avg_reviews：同發行商過去作品平均 num_reviews_total
#
# 因一款遊戲可能有多位發行商，先 explode() 展開再聚合回 appid。
# 若發行商沒有過去作品：平均值用全局平均補，作品數用 0。
def compute_publisher_features(df):
    required_cols = ["appid", "release_date", "publishers", "user_rating", "num_reviews_total"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for publisher history features: {missing}")

    tmp = df[required_cols].explode("publishers")
    tmp = tmp.sort_values(["publishers", "release_date"])

    g = tmp.groupby("publishers", sort=False)

    tmp["pub_game_count_past"] = g.cumcount()

    tmp["pub_rating_sum_past"] = g["user_rating"].cumsum() - tmp["user_rating"]
    tmp["pub_reviews_sum_past"] = g["num_reviews_total"].cumsum() - tmp["num_reviews_total"]

    tmp["publisher_score"] = tmp["pub_rating_sum_past"] / tmp["pub_game_count_past"]
    tmp["publisher_avg_reviews"] = tmp["pub_reviews_sum_past"] / tmp["pub_game_count_past"]

    global_rating = df["user_rating"].mean()
    global_reviews = df["num_reviews_total"].mean()

    tmp["publisher_score"] = tmp["publisher_score"].fillna(global_rating)
    tmp["publisher_avg_reviews"] = tmp["publisher_avg_reviews"].fillna(global_reviews)

    pub_agg = (
        tmp.groupby("appid")
           .agg(
               publisher_score=("publisher_score", "mean"),
               publisher_game_count=("pub_game_count_past", "mean"),
               publisher_avg_reviews=("publisher_avg_reviews", "mean"),
           )
    )

    df = df.merge(pub_agg, on="appid", how="left")
    return df


# -----------------------------------------------------
# 5. Merge wishlist features
# -----------------------------------------------------
# Merge wishlist-based features (rank, followers) into the main dataset by appid.
# Use left join so games not in the wishlist file remain in the dataset.
# Fill missing wishlist values with sensible defaults:
# - wishlist_rank: max_rank + 1 (treated as "very low rank")
# - wishlist_followers: 0
#
# 合併願望清單相關特徵（排名、追蹤人數）到主資料集（以 appid 對應）。
# 使用 left join，確保不在清單內的遊戲仍保留在主資料集中。
# 缺失值處理：
# - wishlist_rank：以 max_rank + 1 填補（視為排名很後面）
# - wishlist_followers：以 0 填補
def add_wishlist_features(df, filename="wishlists_top1000.csv"):
    wishlist_path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(wishlist_path):
        print(f"Wishlist file not found, skip merge: {wishlist_path}")
        print(f"Wishlist file not found, skip merge (Chinese): {wishlist_path} / 找不到願望清單檔案，略過合併步驟")
        return df

    w = pd.read_csv(wishlist_path)

    if "appid" not in w.columns:
        print("Wishlist file has no 'appid' column, skip merge")
        print("Wishlist file has no 'appid' column, skip merge (Chinese) / 願望清單檔案缺少 'appid' 欄位，略過合併")
        return df

    try:
        w["appid"] = w["appid"].astype(df["appid"].dtype)
    except Exception:
        w["appid"] = w["appid"].astype(str)
        df["appid"] = df["appid"].astype(str)

    use_cols = ["appid"]
    if "rank" in w.columns:
        use_cols.append("rank")
    if "followers" in w.columns:
        use_cols.append("followers")

    w = w[use_cols]
    df = df.merge(w, on="appid", how="left")

    rename_map = {}
    if "rank" in df.columns:
        rename_map["rank"] = "wishlist_rank"
    if "followers" in df.columns:
        rename_map["followers"] = "wishlist_followers"
    df = df.rename(columns=rename_map)

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
# - num_lang: number of supported text languages
# - num_audio_lang: number of supported full audio languages
# - One-hot encode top-N most common languages for both text and audio
#
# 建立語言相關特徵，包括：
# - num_lang：支援文字語言數量
# - num_audio_lang：支援語音語言數量
# - 取最常見前 N 種語言做 one-hot（文字與語音分開）
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
# Each unique value becomes a binary feature column.
#
# 將多類別（如 genres、tags）的欄位轉換成 multi-hot 向量。
# 每個唯一值都會變成一個二元特徵欄位。
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
# - year, month, weekday
# - is_weekend: 1 if Saturday/Sunday else 0
# - quarter, season
#
# 從發售日期萃取時間特徵：
# - 年、月、星期幾
# - is_weekend：週末（六日）為 1，否則為 0
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
# success_level is computed from peak_ccu using thresholds:
# - 0: peak_ccu < low
# - 1: low <= peak_ccu < high
# - 2: peak_ccu >= high
#
# 依照 peak_ccu 的門檻將遊戲分類為：
# - 0：peak_ccu < low
# - 1：low <= peak_ccu < high
# - 2：peak_ccu >= high
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
# End-to-end preprocessing pipeline:
# 1) Load raw dataset
# 2) Parse list-like columns into Python lists
# 3) Create basic pre-release features (name length, language, date, tags/genres)
# 4) Create historical vendor features (developer/publisher past performance only)
# 5) Merge wishlist features (optional supplementary dataset)
# 6) Create label success_level from peak_ccu
# 7) Filter out extremely low peak_ccu if needed
# 8) Assemble final feature columns and export to CSV
#
# 完整前處理流程：
# 1) 讀取原始資料
# 2) 將 list 型欄位從字串轉成 Python list
# 3) 建立發售前可得的基本特徵（名稱長度、語言、日期、標籤/類型）
# 4) 建立廠商履歷特徵（只用過去作品，避免洩漏）
# 5) 合併願望清單特徵（可選的副資料集）
# 6) 由 peak_ccu 產生 success_level 標籤
# 7) 視需求移除 peak_ccu 過低的樣本
# 8) 組合最終特徵並輸出 CSV
def preprocess():
    df = load_dataset()

    # Ensure user_rating exists; if missing, compute from positive/negative.
    # Note: user_rating is treated as an outcome-type metric for the current game,
    # so it will NOT be exported as an X feature; it is only used to build vendor history features.
    #
    # 確保 user_rating 存在；若缺少則用 positive/negative 計算。
    # 注意：user_rating 對「本作」屬於結果型欄位，不會輸出成 X 特徵，只用來計算廠商歷史履歷特徵。
    if "user_rating" not in df.columns:
        df["user_rating"] = df["positive"] / (df["positive"] + df["negative"]) * 100

    list_cols = [
        "supported_languages", "full_audio_languages",
        "developers", "publishers",
        "categories", "genres", "tags"
    ]
    df = parse_list_columns(df, list_cols)

    # Name-based feature: length of the game title
    # 名稱特徵：遊戲名稱長度
    df["name_length"] = df["name"].astype(str).apply(len)

    # Pre-release features
    # 發售前可得特徵
    df = build_language_features(df, top_n=10)
    df = build_date_features(df)

    # Vendor historical features (no leakage)
    # 廠商歷史履歷特徵（不洩漏）
    df = compute_developer_score(df)
    df = compute_publisher_features(df)

    # Supplementary dataset (wishlist)
    # 副資料集（願望清單）
    df = add_wishlist_features(df, filename="wishlists_top1000.csv")

    # High-cardinality multi-hot encoding
    # 高基數 multi-hot 編碼
    df = encode_multi_hot(df, "genres", "genre")
    df = encode_multi_hot(df, "tags", "tag")

    # Target label
    # 目標標籤
    df = compute_success_level(df, low=500, high=5000)

    # Filter out extremely low-CCU games
    # 移除 peak_ccu 過低的樣本（視需求）
    if "peak_ccu" in df.columns:
        before_len = len(df)
        df = df[df["peak_ccu"] >= 100].copy()
        after_len = len(df)
        print(f"Removed {before_len - after_len} rows with peak_ccu < 100")
    else:
        print("peak_ccu not found; skipping filtering step")

    # Drop label helper columns if present
    # 若存在衍生/多餘欄位則移除
    if "popularity" in df.columns:
        df = df.drop(columns=["popularity"])

    # Final exported features:
    # - Keep only pre-release obtainable features and vendor history features
    # - Do NOT export current-game outcome features (reviews/ratings/etc.)
    #
    # 最終輸出特徵：
    # - 僅保留發售前可得 + 廠商歷史履歷特徵
    # - 不輸出本作結果型欄位（評論數/評價/推薦數等）
    BASE_FEATURES = [
        "appid",
        "name",
        "price",
        "windows", "mac", "linux",
        "name_length",
        "num_lang", "num_audio_lang",

        # Developer history features
        # 開發商履歷特徵
        "developer_score",
        "developer_game_count",
        "developer_avg_reviews",
        "developer_avg_recommendations",

        # Publisher history features
        # 發行商履歷特徵
        "publisher_score",
        "publisher_game_count",
        "publisher_avg_reviews",

        # Release timing features
        # 發售時間特徵
        "release_year", "release_month", "release_dayofweek",
        "is_weekend", "release_quarter", "release_season",

        # Wishlist features (optional supplementary dataset)
        # 願望清單特徵（可選副資料集）
        "wishlist_rank", "wishlist_followers",
    ]

    GENRE_FEATURES = [c for c in df.columns if c.startswith("genre_")]
    TAG_FEATURES = [c for c in df.columns if c.startswith("tag_")]
    LANG_FEATURES = [c for c in df.columns if c.startswith("lang_") or c.startswith("audio_lang_")]

    OUTPUT_COLS = BASE_FEATURES + LANG_FEATURES + GENRE_FEATURES + TAG_FEATURES + ["success_level"]
    OUTPUT_COLS = list(dict.fromkeys(OUTPUT_COLS))
    df_final = df[OUTPUT_COLS]

    #Save full dataset
    df_final.to_csv(OUTPUT_FILE_FULL, index=False)
    

    #save main dataset
    cols_main = [c for c in df_final.columns if c not in ["wishlist_rank", "wishlist_followers"]]
    df_main = df_final[cols_main]
    df_main.to_csv(OUTPUT_FILE_MAIN, index=False)

    print("Preprocessing completed")
    print(f"1. {os.path.basename(OUTPUT_FILE_FULL)} (Rows: {len(df_final)})")
    print(f"2. {os.path.basename(OUTPUT_FILE_MAIN)} (Rows: {len(df_main)})")
    print(f"Total rows: {len(df_final)}")
    print(f"Number of features (excluding success_level): {len(df_final.columns) - 1}")
    print("success_level distribution:")
    print(df_final["success_level"].value_counts().sort_index())


if __name__ == "__main__":
    preprocess()