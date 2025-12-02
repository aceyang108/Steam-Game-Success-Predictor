import os
from pathlib import Path

import numpy as np
import pandas as pd


# 專案目錄與路徑設定（假設本檔案放在 src/ 底下）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

RAW_FILENAME = "games_march2025_cleaned.csv"


def load_raw_data() -> pd.DataFrame:
    """讀取原始遊戲資料"""
    csv_path = os.path.join(RAW_DATA_PATH, RAW_FILENAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始資料檔案：{csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["release_date"])
    print(f"[load_raw_data] raw shape: {df.shape}")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """依照 name + release_date 去除重複遊戲"""
    before = df.shape[0]
    df = df.drop_duplicates(subset=["name", "release_date"])
    after = df.shape[0]
    print(f"[drop_duplicates] dropped {before - after} duplicates, new shape: {df.shape}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """丟掉用不到或缺值太多的欄位（跟 notebook 內容一致）"""

    columns_to_drop = [
        # Kinda pointless:
        "required_age",
        "dlc_count",
        "header_image",
        "website",
        "support_url",
        "support_email",
        "metacritic_url",
        "achievements",
        "notes",
        "packages",
        "screenshots",
        "movies",
        # Text descriptions (暫時不用文字特徵):
        "detailed_description",
        "about_the_game",
        "short_description",
        "reviews",
        # Too many empty values:
        "metacritic_score",
        "user_score",
        "score_rank",
        "average_playtime_forever",
        "median_playtime_forever",
        # 不考慮近期活躍度:
        "average_playtime_2weeks",
        "median_playtime_2weeks",
        "discount",
        "pct_pos_recent",
        "num_reviews_recent",
    ]

    # 僅丟掉真的存在的欄位，避免未來 schema 變動時炸掉
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=columns_to_drop)
    print(f"[drop_unused_columns] dropped {len(columns_to_drop)} columns, new shape: {df.shape}")
    return df


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    篩掉：
    - peak_ccu == 0
    - positive == 0 且 negative == 0（完全沒評價）
    """

    if not {"peak_ccu", "positive", "negative"}.issubset(df.columns):
        raise KeyError("資料中缺少 peak_ccu / positive / negative 欄位，無法篩選無效資料。")

    drop_condition = (df["peak_ccu"] == 0) | (
        (df["positive"] == 0) & (df["negative"] == 0)
    )

    before = df.shape[0]
    df = df.loc[~drop_condition].copy()
    after = df.shape[0]
    print(f"[filter_invalid_rows] dropped {before - after} rows, new shape: {df.shape}")
    return df


def add_user_rating(df: pd.DataFrame) -> pd.DataFrame:
    """根據 positive / negative 計算 user_rating（百分比）"""

    denom = df["positive"] + df["negative"]
    # 依照前面 filter 的條件，分母理論上不會是 0
    df["user_rating"] = df["positive"] / denom * 100
    print("[add_user_rating] added column: user_rating")
    return df


def add_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 estimated_owners + peak_ccu 計算 owners_score / ccu_score / total_score / popularity
    popularity 只當分析用，不會丟到模型特徵裡。
    """

    if "estimated_owners" not in df.columns:
        raise KeyError("資料中缺少 estimated_owners 欄位，無法計算 owners_score。")
    if "peak_ccu" not in df.columns:
        raise KeyError("資料中缺少 peak_ccu 欄位，無法計算 ccu_score。")

    owners_score_map = {
        "0 - 20000": -3,
        "20000 - 50000": -1,
        "50000 - 100000": 1,
        "100000 - 200000": 3,
        "200000 - 500000": 5,
        "500000 - 1000000": 7,
        "1000000 - 2000000": 9,
        "2000000 - 5000000": 11,
        "5000000 - 10000000": 13,
        "10000000 - 20000000": 17,
        "20000000 - 50000000": 19,
        "50000000 - 100000000": 21,
    }

    df["owners_score"] = df["estimated_owners"].map(owners_score_map)

    unknown_mask = df["owners_score"].isna()
    if unknown_mask.any():
        print("[add_popularity] Warning: unknown estimated_owners values:")
        print(df.loc[unknown_mask, "estimated_owners"].value_counts())

    # 對未知區間先給較高分（之後可再調整）
    df["owners_score"] = df["owners_score"].fillna(23)

    # 依 peak_ccu 分群
    ccu_bins = [
        df["peak_ccu"] < 1000,
        (df["peak_ccu"] >= 1000) & (df["peak_ccu"] < 2500),
        (df["peak_ccu"] >= 2500) & (df["peak_ccu"] < 5000),
        (df["peak_ccu"] >= 5000) & (df["peak_ccu"] < 10000),
        (df["peak_ccu"] >= 10000) & (df["peak_ccu"] < 50000),
        (df["peak_ccu"] >= 50000) & (df["peak_ccu"] < 100000),
        (df["peak_ccu"] >= 100000) & (df["peak_ccu"] < 200000),
        (df["peak_ccu"] >= 200000) & (df["peak_ccu"] < 500000),
    ]
    ccu_scores = [-3, -1, 1, 3, 5, 7, 11, 13]

    df["ccu_score"] = np.select(ccu_bins, ccu_scores, default=15)
    df["total_score"] = df["owners_score"] + df["ccu_score"]

    rating_bins = [
        df["total_score"] <= 0,
        (df["total_score"] > 0) & (df["total_score"] <= 5),
        (df["total_score"] > 5) & (df["total_score"] <= 17),
        (df["total_score"] > 17) & (df["total_score"] <= 27),
    ]
    ratings = ["FAILED", "NICHE", "AVERAGE", "POPULAR"]

    df["popularity"] = np.select(rating_bins, ratings, default="PHENOMENAL")

    print("[add_popularity] added columns: owners_score, ccu_score, total_score, popularity")
    return df


def add_success_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 peak_ccu 定義 success_level:
        0 = 冷門：peak_ccu < 500
        1 = 普通：500 <= peak_ccu < 5000
        2 = 熱門：peak_ccu >= 5000
    """

    if "peak_ccu" not in df.columns:
        raise KeyError("資料中缺少 peak_ccu 欄位，無法產生 success_level。")

    low_threshold = 500
    high_threshold = 5000

    def map_success(peak_ccu: float) -> int:
        if peak_ccu < low_threshold:
            return 0
        elif peak_ccu < high_threshold:
            return 1
        else:
            return 2

    df["success_level"] = df["peak_ccu"].apply(map_success).astype(int)
    print("[add_success_level] success_level value counts:")
    print(df["success_level"].value_counts().sort_index())
    return df


def select_numeric_and_drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 丟掉只用來計算 popularity 的中間分數欄位
    - 僅保留數值欄位
    - 排除會造成 target leakage 的 peak_ccu
    - 輸出 DataFrame 作為 training_data.csv
    """

    # 1) 丟掉中間用的 scoring 欄位
    cols_to_drop_scores = ["owners_score", "ccu_score", "total_score"]
    cols_to_drop_scores = [c for c in cols_to_drop_scores if c in df.columns]
    df = df.drop(columns=cols_to_drop_scores)
    print(f"[select_numeric_and_drop_leakage] dropped score cols: {cols_to_drop_scores}")

    # 2) 只取數值欄位
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 3) 為了避免 target leakage，不把用來切 success_level 的 peak_ccu 當特徵
    leak_cols = ["peak_ccu"]
    numeric_cols = [c for c in numeric_cols if c not in leak_cols]

    df_final = df[numeric_cols].copy()
    print(f"[select_numeric_and_drop_leakage] final numeric columns: {numeric_cols}")

    # ✅ 提醒：training 時請記得把 appid 從 X 裡排除，它只是識別碼
    return df_final


def main():
    df = load_raw_data()
    df = drop_duplicates(df)
    df = drop_unused_columns(df)
    df = filter_invalid_rows(df)
    df = add_user_rating(df)
    df = add_popularity(df)
    df = add_success_level(df)

    df_final = select_numeric_and_drop_leakage(df)

    save_path = os.path.join(PROCESSED_DATA_PATH, "training_data.csv")
    df_final.to_csv(save_path, index=False)
    print(f"[main] Saved training data to: {save_path}")


if __name__ == "__main__":
    main()