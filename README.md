# 🎮 Steam 遊戲首週熱度預測系統 (Steam Game Success Predictor)

## 1. 專案簡介 (Project Overview)
本專案旨在利用機器學習模型，根據 Steam 遊戲在「發售前」的公開資訊（如價格、標籤、發行商），預測遊戲的市場潛力上限(Market Potential / Max Reach)。

我們希望透過此模型幫助獨立遊戲開發者了解：
* **關鍵影響因子：** 哪些特徵（Feature）最能影響遊戲熱度？
* **最佳發售時機：** 選擇在什麼時間點（Timing）發售能避開競爭，獲得最大流量？

---

## 2. 資料集架構 (Dataset Architecture)

本專案使用「主 + 副」雙層資料結構，以確保模型能同時利用 *遊戲內部特徵* 與 *市場外部環境*。

---

### 2.1 Major Dataset（主資料集）

來源：Kaggle + SteamDB

內容包含遊戲本體資訊：

| 類別 | 說明 |
| --- | --- |
| 基礎屬性 | appid、名稱、發售日期、價格、平台支援 (Win/Mac/Linux) |
| 社群評價 | positive / negative 數、user_rating（自動計算） |
| 多國語言資訊 | supported_languages、audio_languages |
| 類別描述 | tags、genres、categories |
| 開發 / 發行資訊 | developers、publishers |
| Steam 使用者行為 | peak_ccu (用於產生 success_level) |

---

### 主資料集特徵（X）

現在的 preprocessing.py 輸出特徵包括：

- 遊戲基本特徵：price、platforms、name_length
- 語言相關：num_lang、num_audio_lang、多語言 one-hot
- 開發商歷史評價：developer_score
- 發行商歷史評價：publisher_score、publisher_game_count
- 評價資訊：positive、negative、user_rating、pct_pos_total
- 日期特徵：release_year、month、weekday、season、quarter
- 類別特徵：genres multi-hot
- 標籤特徵：tags multi-hot
- 心願單相關特徵：wishlist_rank、wishlist_followers

---

### 預測目標（Y）

使用 peak_ccu 分類成：

| success_level | 意義 |
| --- | --- |
| 0 | 冷門 (Cold) |
| 1 | 普通 (Normal) |
| 2 | 熱門 (Hot) |

分類門檻目前為：

- `100 ~ 499` → 0
- `500 ~ 4999` → 1
- `>= 5000` → 2

---

### 2.2 Supplementary Dataset（副資料集）

目前副資料集來源：

### (A) SteamDB Wishlist（願望清單）

包含：

- wishlist_rank
- wishlist_followers

系統會自動整合成特徵：

- `wishlist_rank`（越小越熱門）
- `wishlist_followers`（願望清單追蹤量）

缺失處理方式：

- rank 缺失 → 使用 max_rank + 1
- followers 缺失 → 0

### (B) Steam Calendar（未來可加入）

目前程式中架構已支援，但暫未使用：

- is_sale_period（是否撞 Steam Sale）
- is_holiday_season（旺季）

可於未來加入模型改善表現。

---

## 3. 技術堆疊 (Tech Stack)
我們選擇適合表格數據 (Tabular Data) 且具備高可解釋性的技術方案。

* **語言：** `Python`
* **核心模型：** `XGBoost` / `Random Forest` (處理結構化數據表現最佳)
* **資料處理：** `Pandas`, `NumPy`
* **可解釋性 AI (XAI)：** `SHAP` (用來解釋為什麼模型認為某款遊戲會紅)
* **互動展示：** `Streamlit` (建構 Web App 供使用者動態輸入參數並查看預測結果)

---

## 4. 資料準備 (Data Setup)

1.  **下載主資料集 (Major Dataset):**
    * 前往下載： https://www.kaggle.com/datasets/artermiloff/steam-games-dataset(2025 2024Peak ccu)
                https://www.kaggle.com/datasets/lucaortolan/top-5000-steam-games(2years ago)
                https://www.kaggle.com/datasets/fronkongames/steam-games-dataset(dataset不太正確，僅供參考)
                https://www.youtube.com/watch?v=Id2iYV3EfG4
    * 下載後解壓縮，找到包含遊戲數據的主 CSV 檔。
    * **重新命名 (Rename):** 將檔案改名為 `steam_games.csv`
    * **移動檔案 (Move):** 將檔案放入 `data/raw/` 資料夾中。

2.  **確認副資料集 (Supplementary Dataset):**
    * 找到steam sale calender。
    * 確認 `data/raw/steam_games.csv` 
    * 確認 `data/raw/steam_calendar.csv` 

3.  **完成後目錄結構應如下：**
    ```text
    Steam-Game-Success-Predictor/
    │
    ├── data/
    │   ├── raw/
    │   │   └── games_march2025_cleaned.csv
    │   │
    │   └── processed/
    │       ├── data_after_preprocessing.csv
    │       ├── training_data.csv
    │       ├── wishlists_top1000.csv
    │       └── wishlists_upcoming.csv
    │
    ├── models/
    │   └── xgb_model.json
    │
    ├── src/
    │   ├── make_csv.ipynb
    │   ├── preprocessing.py
    │   ├── steamdb_crawler.py
    │   └── train.py
    │
    ├── .gitignore
    ├── environment.yml
    ├── README.md
    └── requirement.txt
    ```

### Preprocessing pipeline

執行：

```
python src/preprocessing.py

```

輸出：

```
data/processed/data_after_preprocessing.csv

```

---

## 5. 訓練模型 (Training)

執行：

```
python src/train.py

```

輸出：

- 混淆矩陣
- classification report
- cross-validation 結果
- 模型檔案：`models/xgb_model.json`

---

## 6. 專案時程 (3-Week Roadmap)

### 📅 Week 1: 基礎建設與資料清洗 (Baseline)
- [ ] **Data:** 下載 Kaggle 資料集，並建立「特賣會日期表 (Supplementary)」。
- [ ] **Data:** 調整preprocessing.py到配合Kaggle 資料集。
- [ ] **Model:** 定義 Y 的分類門檻，跑通第一個 XGBoost 模型，取得 Baseline 準確率。

### 📅 Week 2: 模型優化與可解釋性 (Optimization & XAI)
- [ ] **Feature:** 將副資料集特徵 (`is_sale_period`) 加入訓練，比較準確率是否提升，還有時間可加入(重大更新日期...)。
- [ ] **XAI:** 實作 SHAP Analysis，產出特徵重要性圖表 (Feature Importance Plot)。
- [ ] **Analysis:** 分析並記錄有趣發現 (例如：Indie 遊戲是否特別害怕撞期夏特？)。

### 📅 Week 3: 互動展示與報告 (Demo & Presentation)
- [ ] **Demo:** 使用 `Streamlit` 製作預測介面 (輸入遊戲資訊 -> 輸出預測等級)。
- [ ] **Report:** 撰寫期末報告，強調「資料來源的邏輯 (Major vs Supplementary)」與「模型解釋 (SHAP)」。
- [ ] **Final:** 準備 PPT 與 Live Demo 腳本。