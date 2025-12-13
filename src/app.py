import os
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import altair as alt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models")

DATA_FILE = os.path.join(PROCESSED_DATA_PATH, "data_after_preprocessing.csv")
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.json")


# -------------------------
# Cache loaders
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model


# -------------------------
# Helpers
# -------------------------
LABEL_ORDER = ["Cold (0)", "Normal (1)", "Hot (2)"]  # ✅ 全站統一順序
LABEL_MAP = {0: "Cold", 1: "Normal", 2: "Hot"}


def decode_label(y: int) -> str:
    return LABEL_MAP.get(int(y), "Unknown")


def build_empty_feature_row(feature_cols):
    """
    Build a 1-row DataFrame with all training features initialized to 0.0.

    建立一列全為 0 的特徵列，欄位與訓練時 feature_cols 完全一致。
    """
    return pd.DataFrame([{c: 0.0 for c in feature_cols}], columns=feature_cols)


def set_if_exists(x_df, col, value):
    """
    Set value to column if the feature column exists.

    若欄位存在於特徵集合中，則寫入指定值。
    """
    if col in x_df.columns:
        x_df.loc[0, col] = value


def prepare_xy(df: pd.DataFrame):
    """
    Prepare X/y exactly like train.py:
    - y = success_level (int)
    - X = drop success_level, drop appid if exists
    - keep numeric only
    """
    if "success_level" not in df.columns:
        raise ValueError("Column 'success_level' not found in data.")

    y = df["success_level"].astype(int)
    X = df.drop(columns=["success_level"])

    if "appid" in X.columns:
        X = X.drop(columns=["appid"])

    X = X.select_dtypes(include=["number"])
    if X.shape[1] == 0:
        raise ValueError("No numeric features found after preprocessing.")
    return X, y


def build_x_from_row(row: pd.Series, feature_cols: pd.Index) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to training features:
    - drop non-feature columns (success_level, appid, name)
    - reindex(columns=feature_cols)
    - fillna(0), astype(float)
    """
    x = row.to_frame().T

    for col in ["success_level", "appid", "name"]:
        if col in x.columns:
            x = x.drop(columns=[col])

    x = x.reindex(columns=feature_cols)
    x = x.fillna(0.0).astype(float)
    return x


def predict_one(model, x_df: pd.DataFrame):
    x_input = x_df.values
    y_pred = int(model.predict(x_input)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_input)[0]
    return y_pred, proba


def render_prediction_panel(y_pred: int, proba):
    """
    Render prediction panel in a user-friendly style.

    以一般使用者能看懂的方式呈現預測結果（統一版面）。
    """
    label = decode_label(y_pred)

    if y_pred == 2:
        badge_class = "badge badge-hot"
        headline = "高熱度（Hot）"
        hint = "模型推估：首週較有機會形成高峰流量。"
        emoji = "🔥"
    elif y_pred == 1:
        badge_class = "badge badge-normal"
        headline = "中等熱度（Normal）"
        hint = "模型推估：熱度落在一般區間，仍有機會靠行銷/口碑拉升。"
        emoji = "✅"
    else:
        badge_class = "badge badge-cold"
        headline = "低熱度（Cold）"
        hint = "模型推估：首週爆發力較弱，較依賴曝光策略與選檔期。"
        emoji = "❄️"

    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
            <div style="flex:1;">
              <div class="{badge_class}">預測結果</div>
              <div style="font-size:28px; font-weight:900; margin-top:10px; color: var(--text);">{emoji} {headline}</div>
              <div class="muted" style="margin-top:8px; line-height:1.6;">{hint}</div>
            </div>
            <div style="width:240px; text-align:right;">
              <div class="muted">Label</div>
              <div style="font-size:16px; font-weight:900; color: var(--text);">{label} ({y_pred})</div>
              <div class="divider" style="margin:10px 0 10px 0;"></div>
              <div class="muted">信心提示</div>
              <div style="font-size:14px; color: var(--subtext);">
                {("可提供機率分佈" if proba is not None else "此模型未提供機率")}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if proba is not None:
        # ✅ 統一：Cold → Normal → Hot
        c1, c2, c3 = st.columns(3)
        c1.metric("Cold (0)", f"{proba[0]*100:.1f}%")
        c2.metric("Normal (1)", f"{proba[1]*100:.1f}%")
        c3.metric("Hot (2)", f"{proba[2]*100:.1f}%")

        proba_df = pd.DataFrame(
            {"Class": LABEL_ORDER, "Probability": [proba[0], proba[1], proba[2]]}
        )

        chart = (
            alt.Chart(proba_df)
            .mark_bar()
            .encode(
                x=alt.X("Class:N", sort=LABEL_ORDER, title=None),
                y=alt.Y("Probability:Q", title=None),
                tooltip=["Class:N", alt.Tooltip("Probability:Q", format=".2%")],
            )
            .properties(height=220)
            .configure_view(strokeOpacity=0)
            .configure(background="transparent")
            .configure_axis(
                labelColor="#111827",
                titleColor="#111827",
                gridColor="rgba(17,24,39,0.10)",
                domainColor="rgba(17,24,39,0.18)",
                tickColor="rgba(17,24,39,0.18)",
            )
        )
        st.altair_chart(chart, use_container_width=True)


# -------------------------
# Main UI
# -------------------------
def main():
    st.set_page_config(page_title="Steam 遊戲首週熱度預測系統", layout="wide")

    # --- Session state for keeping last prediction ---
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = None

    # ✅ 改成淺色系（避免白底白字）
    st.markdown(
        """
        <style>
        :root{
          --bg: #f6f7fb;
          --panel: #ffffff;
          --panel-2: #ffffff;
          --border: rgba(17,24,39,0.12);
          --text: #111827;
          --subtext: rgba(17,24,39,0.72);
          --muted: rgba(17,24,39,0.60);
          --accent: #2563eb;
          --accent-2: #4f46e5;
        }

        html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
        [data-testid="stHeader"] { background: transparent !important; }

        /* 不要全域強制白字，改為整體深色字 */
        h1, h2, h3, h4, h5, p, span, div { color: var(--text); }

        .block-container { padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1200px; }

        .hero {
            padding: 20px 20px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(79,70,229,0.10));
            border: 1px solid var(--border);
            margin-bottom: 14px;
        }
        .hero-title { font-size: 32px; font-weight: 900; line-height: 1.1; color: var(--text); }
        .hero-sub { color: var(--subtext); font-size: 14.5px; margin-top: 8px; line-height: 1.6; }

        .card {
            padding: 16px 16px;
            border-radius: 16px;
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: 0 6px 18px rgba(17,24,39,0.06);
        }
        .card-tight {
            padding: 14px 14px;
            border-radius: 16px;
            background: var(--panel-2);
            border: 1px solid var(--border);
        }

        .muted { color: var(--muted); font-size: 13px; }
        .divider { height: 1px; background: rgba(17,24,39,0.10); margin: 14px 0; }

        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 900;
            border: 1px solid var(--border);
            background: rgba(17,24,39,0.04);
            color: var(--text);
        }
        .badge-hot { background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.25); }
        .badge-normal { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.25); }
        .badge-cold { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.25); }

        /* Sidebar：改亮一些，提高可讀性 */
        [data-testid="stSidebar"] {
          background: #0b1220 !important;
          border-right: 1px solid rgba(148,163,184,0.18);
        }
        [data-testid="stSidebar"] * { color: #e8eefc !important; }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
          color: #e8eefc !important;
        }

        /* 讓 Tab/輸入元件在淺色主畫面保持一致 */
        [data-baseweb="tab"] { color: var(--text) !important; }
        [data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }

        /* 避免某些元件出現白底白字/灰到看不到 */
        .stTextInput input, .stNumberInput input, .stSelectbox div, .stMultiSelect div {
          color: var(--text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">🎮 Steam 遊戲首週熱度預測系統</div>
          <div class="hero-sub">
            使用「發售前可得資訊」＋「廠商過往履歷」推估遊戲熱度等級（Cold / Normal / Hot）。<br/>
            <span style="color: rgba(17,24,39,0.62);">提示：此為統計模型推估結果，非保證玩家數或銷量。</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load
    try:
        df = load_data()
        model = load_model()
        X_all, y_all = prepare_xy(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    feature_cols = X_all.columns

    # Feature groups (for advanced UI)
    lang_cols = [c for c in feature_cols if c.startswith("lang_")]
    audio_lang_cols = [c for c in feature_cols if c.startswith("audio_lang_")]
    genre_cols = [c for c in feature_cols if c.startswith("genre_")]
    tag_cols = [c for c in feature_cols if c.startswith("tag_")]

    # Sidebar
    st.sidebar.header("操作流程")
    mode = st.sidebar.selectbox(
        "① 選擇模式",
        ["資料庫遊戲（直接預測）", "情境試算（調整參數）", "資料庫外遊戲（手動輸入）"],
        index=0,
        help="一般展示用「直接預測」；想看參數影響用「情境試算」；資料庫找不到的遊戲用「手動輸入」。"
    )

    with st.sidebar.expander("資料概況", expanded=False):
        dist = y_all.value_counts().sort_index().rename(index={0: "Cold", 1: "Normal", 2: "Hot"})
        st.write(f"總遊戲數：{len(df)}")
        st.write("類別分布：")
        st.write(dist)
        st.write(f"特徵數：{len(feature_cols)}")

    # Build x_df depending on mode
    x_df = None
    row = None
    true_label = None
    display_name = None

    # -------------------------
    # Mode A/B: Existing / What-if
    # -------------------------
    if mode in ["資料庫遊戲（直接預測）", "情境試算（調整參數）"]:
        if "name" not in df.columns:
            st.error("資料表缺少 name 欄位，無法使用資料庫遊戲模式。")
            st.stop()

        all_names = df["name"].astype(str).fillna("").unique().tolist()

        st.sidebar.subheader("② 選擇遊戲")
        keyword = st.sidebar.text_input("搜尋名稱（可留空）", "")

        if keyword.strip():
            candidates = [n for n in all_names if keyword.lower() in n.lower()]
            if not candidates:
                st.sidebar.warning("找不到符合關鍵字的遊戲，請改用下拉選單。")
                selected_name = st.sidebar.selectbox("選擇遊戲", all_names)
            else:
                selected_name = st.sidebar.selectbox("選擇遊戲", candidates)
        else:
            selected_name = st.sidebar.selectbox("選擇遊戲", all_names)

        picked = df[df["name"].astype(str) == selected_name]
        if picked.empty:
            st.error("你選的遊戲不在資料集中。")
            st.stop()

        row = picked.iloc[0]
        display_name = str(row.get("name", "N/A"))
        x_df = build_x_from_row(row, feature_cols)

        # What-if controls
        if mode == "情境試算（調整參數）":
            st.sidebar.subheader("③ 調整假設（可選）")

            with st.sidebar.expander("常用調整", expanded=True):
                if "price" in x_df.columns:
                    x_df.loc[0, "price"] = st.slider("價格（USD）", 0.0, 80.0, float(x_df["price"].iloc[0]), 0.5)

                if "wishlist_followers" in x_df.columns:
                    base = float(x_df["wishlist_followers"].iloc[0])
                    maxv = float(max(pd.to_numeric(df.get("wishlist_followers", 0), errors="coerce").max(), 100000))
                    x_df.loc[0, "wishlist_followers"] = st.slider("願望單追蹤（followers）", 0.0, maxv, base, 1000.0)

                if "wishlist_rank" in x_df.columns:
                    base = float(x_df["wishlist_rank"].iloc[0])
                    maxv = float(max(pd.to_numeric(df.get("wishlist_rank", 0), errors="coerce").max(), 10000))
                    x_df.loc[0, "wishlist_rank"] = st.slider("願望單排名（越小越好）", 1.0, maxv, base, 1.0)

            with st.sidebar.expander("進階：語言 / 類型 / 標籤（可不填）", expanded=False):
                if len(lang_cols) > 0:
                    chosen_langs = st.multiselect(
                        "文字語言（text）",
                        options=[c.replace("lang_", "") for c in lang_cols],
                        default=[]
                    )
                    for c in lang_cols:
                        x_df.loc[0, c] = 0.0
                    for l in chosen_langs:
                        set_if_exists(x_df, f"lang_{l}", 1.0)
                    set_if_exists(x_df, "num_lang", float(len(chosen_langs)))

                if len(audio_lang_cols) > 0:
                    chosen_audio = st.multiselect(
                        "語音語言（audio）",
                        options=[c.replace("audio_lang_", "") for c in audio_lang_cols],
                        default=[]
                    )
                    for c in audio_lang_cols:
                        x_df.loc[0, c] = 0.0
                    for l in chosen_audio:
                        set_if_exists(x_df, f"audio_lang_{l}", 1.0)
                    set_if_exists(x_df, "num_audio_lang", float(len(chosen_audio)))

                if len(genre_cols) > 0:
                    chosen_genres = st.multiselect(
                        "Genres",
                        options=[c.replace("genre_", "") for c in genre_cols],
                        default=[]
                    )
                    for g in chosen_genres:
                        set_if_exists(x_df, f"genre_{g}", 1.0)

                if len(tag_cols) > 0:
                    chosen_tags = st.multiselect(
                        "Tags（很多，可少選）",
                        options=[c.replace("tag_", "") for c in tag_cols],
                        default=[]
                    )
                    for t in chosen_tags:
                        set_if_exists(x_df, f"tag_{t}", 1.0)

        if "success_level" in row.index and pd.notna(row["success_level"]):
            true_label = int(row["success_level"])

    # -------------------------
    # Mode C: New Game manual input (simplified)
    # -------------------------
    else:
        st.sidebar.subheader("② 輸入新遊戲資料")
        st.sidebar.caption("不知道的欄位可先用預設值，仍可先得到一個模型推估。")

        display_name = st.sidebar.text_input("遊戲名稱（僅顯示用）", "New Game")
        x_df = build_empty_feature_row(feature_cols)

        def mean_or_zero(col):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                return float(s.mean()) if len(s) else 0.0
            return 0.0

        with st.sidebar.expander("基本資訊（建議填）", expanded=True):
            price = st.number_input("價格（USD）", min_value=0.0, value=0.0, step=1.0)
            windows = st.selectbox("Windows", [0, 1], index=1)
            mac = st.selectbox("Mac", [0, 1], index=0)
            linux = st.selectbox("Linux", [0, 1], index=0)

        with st.sidebar.expander("發售時間（建議填）", expanded=False):
            release_year = st.number_input("年", min_value=1970, max_value=2100, value=2025, step=1)
            release_month = st.number_input("月", min_value=1, max_value=12, value=6, step=1)
            release_dayofweek = st.number_input("星期（0=Mon ... 6=Sun）", min_value=0, max_value=6, value=4, step=1)

        with st.sidebar.expander("廠商履歷（可用預設）", expanded=False):
            dev_score = st.number_input("Developer score", value=mean_or_zero("developer_score"))
            dev_game_count = st.number_input("Developer game count", min_value=0.0, value=mean_or_zero("developer_game_count"), step=1.0)
            dev_avg_reviews = st.number_input("Developer avg reviews", min_value=0.0, value=mean_or_zero("developer_avg_reviews"), step=10.0)
            dev_avg_reco = st.number_input("Developer avg recommendations", min_value=0.0, value=mean_or_zero("developer_avg_recommendations"), step=10.0)

            pub_score = st.number_input("Publisher score", value=mean_or_zero("publisher_score"))
            pub_game_count = st.number_input("Publisher game count", min_value=0.0, value=mean_or_zero("publisher_game_count"), step=1.0)
            pub_avg_reviews = st.number_input("Publisher avg reviews", min_value=0.0, value=mean_or_zero("publisher_avg_reviews"), step=10.0)

        with st.sidebar.expander("願望單（可不填）", expanded=False):
            wishlist_followers = st.number_input("Wishlist followers", min_value=0, value=0, step=100)
            wishlist_rank = st.number_input("Wishlist rank", min_value=1, value=int(max(mean_or_zero("wishlist_rank"), 1000)), step=1)

        with st.sidebar.expander("進階：語言 / 類型 / 標籤（可不填）", expanded=False):
            chosen_langs = st.multiselect(
                "文字語言",
                options=[c.replace("lang_", "") for c in lang_cols],
                default=[]
            )
            chosen_audio = st.multiselect(
                "語音語言",
                options=[c.replace("audio_lang_", "") for c in audio_lang_cols],
                default=[]
            )
            chosen_genres = st.multiselect(
                "Genres",
                options=[c.replace("genre_", "") for c in genre_cols],
                default=[]
            )
            chosen_tags = st.multiselect(
                "Tags（很多，可少選）",
                options=[c.replace("tag_", "") for c in tag_cols],
                default=[]
            )

        # Derived timing
        is_weekend = 1 if int(release_dayofweek) in (5, 6) else 0
        release_quarter = (int(release_month) - 1) // 3 + 1

        def month_to_season(m):
            if m in [3, 4, 5]:
                return 1
            if m in [6, 7, 8]:
                return 2
            if m in [9, 10, 11]:
                return 3
            return 4

        release_season = month_to_season(int(release_month))

        # Write features
        set_if_exists(x_df, "price", float(price))
        set_if_exists(x_df, "windows", float(windows))
        set_if_exists(x_df, "mac", float(mac))
        set_if_exists(x_df, "linux", float(linux))
        set_if_exists(x_df, "name_length", float(len(str(display_name))))

        set_if_exists(x_df, "release_year", float(release_year))
        set_if_exists(x_df, "release_month", float(release_month))
        set_if_exists(x_df, "release_dayofweek", float(release_dayofweek))
        set_if_exists(x_df, "is_weekend", float(is_weekend))
        set_if_exists(x_df, "release_quarter", float(release_quarter))
        set_if_exists(x_df, "release_season", float(release_season))

        set_if_exists(x_df, "wishlist_rank", float(wishlist_rank))
        set_if_exists(x_df, "wishlist_followers", float(wishlist_followers))

        set_if_exists(x_df, "developer_score", float(dev_score))
        set_if_exists(x_df, "developer_game_count", float(dev_game_count))
        set_if_exists(x_df, "developer_avg_reviews", float(dev_avg_reviews))
        set_if_exists(x_df, "developer_avg_recommendations", float(dev_avg_reco))

        set_if_exists(x_df, "publisher_score", float(pub_score))
        set_if_exists(x_df, "publisher_game_count", float(pub_game_count))
        set_if_exists(x_df, "publisher_avg_reviews", float(pub_avg_reviews))

        set_if_exists(x_df, "num_lang", float(len(chosen_langs)))
        set_if_exists(x_df, "num_audio_lang", float(len(chosen_audio)))

        for l in chosen_langs:
            set_if_exists(x_df, f"lang_{l}", 1.0)
        for l in chosen_audio:
            set_if_exists(x_df, f"audio_lang_{l}", 1.0)
        for g in chosen_genres:
            set_if_exists(x_df, f"genre_{g}", 1.0)
        for t in chosen_tags:
            set_if_exists(x_df, f"tag_{t}", 1.0)

        true_label = None

    # Final safety
    x_df = x_df.reindex(columns=feature_cols).fillna(0.0).astype(float)

    # -------------------------
    # Main layout
    # -------------------------
    tab1, tab2 = st.tabs(["結果", "進階分析"])

    with tab1:
        left, right = st.columns([1.25, 1])

        with left:
            st.markdown(
                f"""
                <div class="card-tight">
                  <div style="font-size:18px; font-weight:900; color: var(--text);">本次輸入摘要</div>
                  <div class="divider"></div>
                  <div class="muted">遊戲：<b style="color: var(--text);">{display_name if display_name else "N/A"}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                if "price" in x_df.columns:
                    st.metric("價格", f"${float(x_df['price'].iloc[0]):.2f}")
                if row is not None and "appid" in row.index and pd.notna(row["appid"]):
                    try:
                        st.metric("AppID", f"{int(row['appid'])}")
                    except Exception:
                        st.metric("AppID", f"{row['appid']}")
            with c2:
                if "num_lang" in x_df.columns:
                    st.metric("文字語言", f"{int(x_df['num_lang'].iloc[0])}")
                if "num_audio_lang" in x_df.columns:
                    st.metric("語音語言", f"{int(x_df['num_audio_lang'].iloc[0])}")
            with c3:
                if "wishlist_followers" in x_df.columns:
                    st.metric("願望單追蹤", f"{int(x_df['wishlist_followers'].iloc[0])}")
                if "wishlist_rank" in x_df.columns:
                    st.metric("願望單排名", f"{int(x_df['wishlist_rank'].iloc[0])}")

        with right:
            st.markdown(
                """
                <div class="card-tight">
                  <div style="font-size:18px; font-weight:900; color: var(--text);">操作</div>
                  <div class="divider"></div>
                  <div class="muted">按下按鈕後會保留結果；調整參數後可再次預測。</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            if st.button("開始預測", type="primary", use_container_width=True):
                y_pred, proba = predict_one(model, x_df)
                st.session_state.last_pred = (y_pred, proba)
                st.session_state.last_meta = {
                    "display_name": display_name,
                    "mode": mode,
                    "true_label": true_label,
                }

        st.divider()

        if st.session_state.last_pred is not None:
            y_pred, proba = st.session_state.last_pred
            render_prediction_panel(y_pred, proba)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            st.subheader("對照（資料庫內遊戲才有）")
            if true_label is None:
                st.info("這是資料庫外輸入的遊戲，因此沒有 True level 可對照。")
            else:
                st.markdown(
                    f"""
                    <div class="card-tight">
                      <div class="muted">True level（資料庫標籤）</div>
                      <div style="font-size:18px; font-weight:900; color: var(--text);">{decode_label(int(true_label))} ({int(true_label)})</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="card">
                  <div style="font-size:18px; font-weight:900; color: var(--text);">尚未預測</div>
                  <div class="divider"></div>
                  <div class="muted">請先在右側按「開始預測」。</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        non_zero = int((x_df.iloc[0] != 0).sum())

        st.markdown(
            f"""
            <div class="card">
              <div style="font-size:18px; font-weight:900; color: var(--text);">分析摘要</div>
              <div class="divider"></div>
              <div class="muted">非零特徵數：<b style="color: var(--text);">{non_zero}</b> / {x_df.shape[1]}</div>
              <div class="muted">提示：特徵數很大是因為 tags/genres/languages 的 multi-hot 展開。</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        with st.expander("查看非零特徵（最多 200 筆）", expanded=False):
            nz = x_df.T
            nz.columns = ["value"]
            nz = nz[nz["value"] != 0].sort_values("value", ascending=False)
            st.dataframe(nz.head(200), use_container_width=True)

        st.divider()

        st.subheader("全域特徵重要度（Top 20）")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi_df = pd.DataFrame(
                {"feature": feature_cols, "importance": importances}
            ).sort_values("importance", ascending=False).head(20)

            st.dataframe(fi_df, use_container_width=True)

            fi_top = fi_df.copy()
            fi_top["feature"] = fi_top["feature"].astype(str)

            fi_chart = (
                alt.Chart(fi_top)
                .mark_bar()
                .encode(
                    x=alt.X("importance:Q", title=None),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature:N", alt.Tooltip("importance:Q", format=".4f")],
                )
                .properties(height=420)
                .configure_view(strokeOpacity=0)
                .configure(background="transparent")
                .configure_axis(
                    labelColor="#111827",
                    titleColor="#111827",
                    gridColor="rgba(17,24,39,0.10)",
                    domainColor="rgba(17,24,39,0.18)",
                    tickColor="rgba(17,24,39,0.18)",
                )
            )
            st.altair_chart(fi_chart, use_container_width=True)
        else:
            st.info("此模型未提供 feature_importances_。")


if __name__ == "__main__":
    main()