import os
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models")

DATA_FILE = os.path.join(PROCESSED_DATA_PATH, "data_after_preprocessing.csv")
MODEL_FILE = os.path.join(MODEL_PATH, "xgb_model.json")


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model


def decode_label(y: int) -> str:
    mapping = {0: "Cold", 1: "Normal", 2: "Hot"}
    return mapping.get(int(y), "Unknown")


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

    # keep numeric only
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


def render_label_badge(label_int: int, title: str):
    st.markdown(f"**{title}**")
    if label_int == 2:
        st.success(f"{decode_label(label_int)} (2)")
    elif label_int == 1:
        st.info(f"{decode_label(label_int)} (1)")
    else:
        st.warning(f"{decode_label(label_int)} (0)")


def main():
    st.set_page_config(page_title="Steam Game Success Predictor", layout="wide")
    st.title("Steam Game Success Predictor")
    st.caption("Predict game popularity level (Cold / Normal / Hot) based on processed features")

    # Load
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Column checks (only needed for Existing Game mode)
    if "name" not in df.columns:
        st.warning("Column 'name' not found. Existing Game UI will be limited, but New Game mode still works.")

    # Prepare X/y (same as training)
    try:
        X_all, y_all = prepare_xy(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    feature_cols = X_all.columns

    # Feature groups for UI (derived from feature_cols)
    lang_cols = [c for c in feature_cols if c.startswith("lang_")]
    audio_lang_cols = [c for c in feature_cols if c.startswith("audio_lang_")]
    genre_cols = [c for c in feature_cols if c.startswith("genre_")]
    tag_cols = [c for c in feature_cols if c.startswith("tag_")]

    # Sidebar
    st.sidebar.header("Controls")
    mode = st.sidebar.radio(
        "Mode",
        ["Existing Game", "What-if Scenario", "New Game (Not in dataset)"],
        index=0
    )

    # Dataset summary
    with st.sidebar.expander("Dataset summary"):
        st.write(f"Total games: {len(df)}")
        dist = y_all.value_counts().sort_index().rename(index={0: "Cold", 1: "Normal", 2: "Hot"})
        st.write("Label distribution:")
        st.write(dist)
        st.write(f"Feature count: {len(feature_cols)}")

    tab_overview, tab_analysis = st.tabs(["Overview", "Analysis"])

    # -------------------------
    # Build x_df depending on mode
    # -------------------------
    x_df = None
    row = None
    true_label = None
    display_name = None

    # -------- Existing / What-if: pick from dataset --------
    if mode in ["Existing Game", "What-if Scenario"]:
        if "name" not in df.columns:
            st.error("Column 'name' not found in data_after_preprocessing.csv. Cannot use Existing/What-if mode.")
            st.stop()

        all_names = df["name"].astype(str).fillna("").unique().tolist()
        search_keyword = st.sidebar.text_input("Search game name keyword", "")

        if search_keyword.strip():
            candidate_names = [n for n in all_names if search_keyword.lower() in n.lower()]
            if not candidate_names:
                st.sidebar.warning("No game found for this keyword.")
                selected_name = st.sidebar.selectbox("Select a game", all_names)
            else:
                selected_name = st.sidebar.selectbox("Select a game", candidate_names)
        else:
            selected_name = st.sidebar.selectbox("Select a game", all_names)

        picked = df[df["name"].astype(str) == selected_name]
        if picked.empty:
            st.error("Selected game not found in dataset.")
            st.stop()
        row = picked.iloc[0]
        display_name = str(row.get("name", "N/A"))

        x_df = build_x_from_row(row, feature_cols)

        # What-if adjustments
        if mode == "What-if Scenario":
            with st.sidebar.expander("What-if adjustments", expanded=True):
                if "price" in x_df.columns:
                    base_price = float(x_df["price"].iloc[0])
                    new_price = st.slider("Price", 0.0, 80.0, base_price, 0.5)
                    x_df["price"] = new_price

                if "num_lang" in x_df.columns:
                    base_num_lang = int(x_df["num_lang"].iloc[0])
                    new_num_lang = st.slider("Number of supported languages", 0, 60, base_num_lang, 1)
                    x_df["num_lang"] = float(new_num_lang)

                if "wishlist_followers" in x_df.columns:
                    base_followers = float(x_df["wishlist_followers"].iloc[0])
                    max_followers = float(max(df.get("wishlist_followers", pd.Series([0])).max(), 100000))
                    new_followers = st.slider("Wishlist followers", 0.0, max_followers, base_followers, 1000.0)
                    x_df["wishlist_followers"] = new_followers

                if "wishlist_rank" in x_df.columns:
                    base_rank = float(x_df["wishlist_rank"].iloc[0])
                    max_rank = float(max(df.get("wishlist_rank", pd.Series([0])).max(), 10000))
                    new_rank = st.slider("Wishlist rank (smaller is better)", 1.0, max_rank, base_rank, 1.0)
                    x_df["wishlist_rank"] = new_rank

                # optional: toggle some tags/genres/languages (only if you want)
                with st.expander("Toggle tags/genres/languages (optional)", expanded=False):
                    if len(lang_cols) > 0:
                        chosen_langs = st.multiselect(
                            "Supported languages (text)",
                            options=[c.replace("lang_", "") for c in lang_cols],
                            default=[]
                        )
                        for c in lang_cols:
                            x_df.loc[0, c] = 0.0
                        for l in chosen_langs:
                            set_if_exists(x_df, f"lang_{l}", 1.0)
                        if "num_lang" in x_df.columns:
                            x_df.loc[0, "num_lang"] = float(len(chosen_langs))

                    if len(audio_lang_cols) > 0:
                        chosen_audio = st.multiselect(
                            "Supported languages (audio)",
                            options=[c.replace("audio_lang_", "") for c in audio_lang_cols],
                            default=[]
                        )
                        for c in audio_lang_cols:
                            x_df.loc[0, c] = 0.0
                        for l in chosen_audio:
                            set_if_exists(x_df, f"audio_lang_{l}", 1.0)
                        if "num_audio_lang" in x_df.columns:
                            x_df.loc[0, "num_audio_lang"] = float(len(chosen_audio))

        if "success_level" in row.index:
            true_label = int(row["success_level"])

    # -------- New Game: build from manual inputs --------
    if mode == "New Game (Not in dataset)":
        display_name = st.sidebar.text_input("Game name (display only)", "New Game")

        # Start from all-zero feature vector (same order as training)
        x_df = build_empty_feature_row(feature_cols)

        # Defaults (use dataset means if available)
        def mean_or_zero(col):
            if col in df.columns:
                try:
                    return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
                except Exception:
                    return 0.0
            return 0.0

        st.sidebar.subheader("Basic info (pre-release)")
        price = st.sidebar.number_input("Price", min_value=0.0, value=0.0, step=1.0)
        windows = st.sidebar.selectbox("Windows", [0, 1], index=1)
        mac = st.sidebar.selectbox("Mac", [0, 1], index=0)
        linux = st.sidebar.selectbox("Linux", [0, 1], index=0)
        name_length = st.sidebar.number_input("Name length", min_value=0, value=10, step=1)

        st.sidebar.subheader("Release timing")
        release_year = st.sidebar.number_input("Release year", min_value=1970, max_value=2100, value=2025, step=1)
        release_month = st.sidebar.number_input("Release month", min_value=1, max_value=12, value=6, step=1)
        release_dayofweek = st.sidebar.number_input("Release dayofweek (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=4, step=1)

        is_weekend = 1 if int(release_dayofweek) in (5, 6) else 0
        release_quarter = (int(release_month) - 1) // 3 + 1

        def month_to_season(m):
            if m in [3, 4, 5]:
                return 1
            elif m in [6, 7, 8]:
                return 2
            elif m in [9, 10, 11]:
                return 3
            return 4

        release_season = month_to_season(int(release_month))

        st.sidebar.subheader("Wishlist (optional)")
        wishlist_rank = st.sidebar.number_input("Wishlist rank", min_value=1, value=int(max(mean_or_zero("wishlist_rank"), 1000)), step=1)
        wishlist_followers = st.sidebar.number_input("Wishlist followers", min_value=0, value=0, step=100)

        st.sidebar.subheader("Vendor history (manual / defaults)")
        dev_score = st.sidebar.number_input("Developer score", value=mean_or_zero("developer_score"))
        dev_game_count = st.sidebar.number_input("Developer game count", min_value=0.0, value=mean_or_zero("developer_game_count"), step=1.0)
        dev_avg_reviews = st.sidebar.number_input("Developer avg reviews", min_value=0.0, value=mean_or_zero("developer_avg_reviews"), step=10.0)
        dev_avg_reco = st.sidebar.number_input("Developer avg recommendations", min_value=0.0, value=mean_or_zero("developer_avg_recommendations"), step=10.0)

        pub_score = st.sidebar.number_input("Publisher score", value=mean_or_zero("publisher_score"))
        pub_game_count = st.sidebar.number_input("Publisher game count", min_value=0.0, value=mean_or_zero("publisher_game_count"), step=1.0)
        pub_avg_reviews = st.sidebar.number_input("Publisher avg reviews", min_value=0.0, value=mean_or_zero("publisher_avg_reviews"), step=10.0)

        st.sidebar.subheader("Languages / Genres / Tags")
        chosen_langs = st.sidebar.multiselect(
            "Supported languages (text)",
            options=[c.replace("lang_", "") for c in lang_cols],
            default=[]
        )
        chosen_audio = st.sidebar.multiselect(
            "Supported languages (audio)",
            options=[c.replace("audio_lang_", "") for c in audio_lang_cols],
            default=[]
        )
        chosen_genres = st.sidebar.multiselect(
            "Genres (multi-hot)",
            options=[c.replace("genre_", "") for c in genre_cols],
            default=[]
        )
        chosen_tags = st.sidebar.multiselect(
            "Tags (multi-hot)",
            options=[c.replace("tag_", "") for c in tag_cols],
            default=[]
        )

        # Write numeric features (only if they exist)
        set_if_exists(x_df, "price", float(price))
        set_if_exists(x_df, "windows", float(windows))
        set_if_exists(x_df, "mac", float(mac))
        set_if_exists(x_df, "linux", float(linux))
        set_if_exists(x_df, "name_length", float(name_length))

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

        # Language counts
        set_if_exists(x_df, "num_lang", float(len(chosen_langs)))
        set_if_exists(x_df, "num_audio_lang", float(len(chosen_audio)))

        # Multi-hot assignments
        for l in chosen_langs:
            set_if_exists(x_df, f"lang_{l}", 1.0)
        for l in chosen_audio:
            set_if_exists(x_df, f"audio_lang_{l}", 1.0)
        for g in chosen_genres:
            set_if_exists(x_df, f"genre_{g}", 1.0)
        for t in chosen_tags:
            set_if_exists(x_df, f"tag_{t}", 1.0)

        true_label = None  # new game has no true label

    # Final safety: ensure numeric float + correct shape
    x_df = x_df.reindex(columns=feature_cols).fillna(0.0).astype(float)

    if x_df.shape[1] != len(feature_cols):
        st.error(f"Feature shape mismatch: expected {len(feature_cols)}, got {x_df.shape[1]}")
        st.stop()

    # Predict
    y_pred, proba = predict_one(model, x_df)

    # -------------------------
    # Overview
    # -------------------------
    with tab_overview:
        st.subheader("Game Overview")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Name:** {display_name if display_name is not None else 'N/A'}")
            if row is not None and "appid" in row.index and pd.notna(row["appid"]):
                try:
                    st.markdown(f"**AppID:** {int(row['appid'])}")
                except Exception:
                    st.markdown(f"**AppID:** {row['appid']}")
            if "price" in x_df.columns:
                st.markdown(f"**Price:** {float(x_df['price'].iloc[0]):.2f}")
        with c2:
            if "release_year" in x_df.columns:
                st.markdown(f"**Release year:** {int(x_df['release_year'].iloc[0])}")
            if "release_month" in x_df.columns:
                st.markdown(f"**Release month:** {int(x_df['release_month'].iloc[0])}")
            if "is_weekend" in x_df.columns:
                st.markdown(f"**Weekend release:** {'Yes' if int(x_df['is_weekend'].iloc[0]) == 1 else 'No'}")
        with c3:
            if "wishlist_rank" in x_df.columns:
                st.markdown(f"**Wishlist rank:** {int(x_df['wishlist_rank'].iloc[0])}")
            if "wishlist_followers" in x_df.columns:
                st.markdown(f"**Wishlist followers:** {int(x_df['wishlist_followers'].iloc[0])}")

        st.markdown("---")
        st.subheader("Model Prediction")

        p1, p2 = st.columns(2)
        with p1:
            render_label_badge(y_pred, "Predicted level")
        with p2:
            if true_label is None:
                st.markdown("**True level**")
                st.write("N/A")
            else:
                render_label_badge(int(true_label), "True level")

        if proba is not None:
            st.markdown("**Class probabilities**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Cold (0)", f"{proba[0]*100:.1f}%")
            m2.metric("Normal (1)", f"{proba[1]*100:.1f}%")
            m3.metric("Hot (2)", f"{proba[2]*100:.1f}%")

            proba_df = pd.DataFrame(
                {"Class": ["Cold (0)", "Normal (1)", "Hot (2)"], "Probability": proba}
            ).set_index("Class")
            st.bar_chart(proba_df)

        st.caption("Inference uses the same numeric feature set and column order as training.")

    # -------------------------
    # Analysis
    # -------------------------
    with tab_analysis:
        st.subheader("Feature Values (This Sample)")

        non_zero = int((x_df.iloc[0] != 0).sum())
        st.write(f"Non-zero features: {non_zero} / {x_df.shape[1]}")

        show_mode = st.radio("Show features", ["Top non-zero only", "All"], horizontal=True)
        if show_mode == "Top non-zero only":
            nz = x_df.T
            nz.columns = ["value"]
            nz = nz[nz["value"] != 0].sort_values("value", ascending=False)
            st.dataframe(nz.head(200))
        else:
            st.dataframe(x_df.T)

        st.markdown("---")
        st.subheader("Global Feature Importance (Top 20)")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi_df = pd.DataFrame(
                {"feature": feature_cols, "importance": importances}
            ).sort_values("importance", ascending=False).head(20)

            st.dataframe(fi_df)
            st.bar_chart(fi_df.set_index("feature")["importance"])
        else:
            st.info("This model does not expose feature_importances_.")


if __name__ == "__main__":
    main()
