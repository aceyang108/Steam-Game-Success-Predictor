# 🎮 Steam 遊戲首週熱度預測系統 (Steam Game Success Predictor)

## 1. 專案簡介 (Project Overview)
本專案旨在利用機器學習模型，根據 Steam 遊戲在「發售前」的公開資訊（如價格、標籤、發行商），預測遊戲的市場潛力上限(Market Potential / Max Reach)。

我們希望透過此模型幫助獨立遊戲開發者了解：
* **關鍵影響因子：** 哪些特徵（Feature）最能影響遊戲熱度？
* **最佳發售時機：** 選擇在什麼時間點（Timing）發售能避開競爭，獲得最大流量？

---

## 2. 資料集架構 (Dataset Architecture)
本專案嚴格遵循 **"Major + Supplementary"** 的雙層資料架構，以模擬真實世界的變數影響。

### 🔹 Major Dataset (主資料集)：遊戲本體數據
* **來源：** Kaggle (Steam Store Games) + SteamDB
* **角色：** 提供「內部屬性」 (Internal Factors)。
* **內容：**
    * **特徵 (X):** 遊戲名稱、發售日期、價格 (Price)、支援平台/語言、標籤 (Categories/Tags/Genres)、開發/發行商、好評/負評及其比例等。
    * **預測目標 (Y):** 根據歷史最高同時在線人數 (All-time Peak CCU)、以及預計持有者數量（Estimated Owners），綜合出熱門度評分。
    * **標籤定義：** 將熱門度評分離散化為五類：`FAILED（爆死）` / `NICHE（小眾）` / `AVERAGE（一般）` / `POPULAR（熱門）` / `PHENOMENAL（現象級）`。

### 🔸 Supplementary Dataset (副資料集)：願望清單數排名 / 追蹤者數量
* **來源：** SteamDB上 `Most wishlisted games（僅含未發售）`、`Wishlist activity（含所有作品）` 兩頁面
* **角色：** 提供「外部環境影響」 (External Environmental Factors)、補充主資料集未收錄資訊。
* **內容：**
    * 分成「所有作品排名 top1000」、以及「未發售作品綜合排名」兩份資料集。
    * 每份資料集中有 `appid`, `rank（願望清單數排名）`, `name（遊戲名）`, `followers（追蹤者）` 這四項資訊。
* **備註：**
    * 若是使用 Chrome 或 Firefox 瀏覽器，`Wishlist activity` 僅能顯示前1000筆作品，使用 Edge 瀏覽器則無此限制。
    但是 SteamDB 的驗證機制比較嚴格，只能使用 undetected_chromedriver 來繞過，於是在該頁面上我只能抓最多1000筆資料。
    * 至於 `Most wishlisted games` 頁面就沒有這個問題，若我們最終要預測的遊戲以未發售作品為主，那麼使用這個作為副資料集是可行的，只是對於已發售作品的預測沒有任何參考價值。
    * Steam 本身並沒有公開每個作品的願望清單數，對此 SteamDB 使用的方案是顯示「願望清單數排名」以及額外添加「追蹤者數量」用於參考。
    但願望清單排名與追蹤者數量並非完全成正比，所以兩項指標都要用來參考。
* **可嘗試的解決方案：**（若是到時候發現此副資料集作用有限）
    1. 嘗試其他瀏覽器 driver 或設定，讓它能夠繞過網頁驗證機制，並且像 Edge 那樣一次顯示更多作品。
    2. 在 `Wishlist activity` 頁面嘗試各種篩選/排序條件，以顯示盡可能多的作品。
    3. 整合兩資料集，刪除重複元素並重新調整 rank。
    4. 尋找網路上其他人整理好的資料。
* **整合方式：** TBD。

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
    * 前往下載： https://www.kaggle.com/datasets/artermiloff/steam-games-dataset (2025 2024Peak ccu)
                https://www.kaggle.com/datasets/lucaortolan/top-5000-steam-games (2years ago)
                https://www.kaggle.com/datasets/fronkongames/steam-games-dataset (dataset不太正確，僅供參考)
                https://www.youtube.com/watch?v=Id2iYV3EfG4
    * 下載後解壓縮，找到包含遊戲數據的主 CSV 檔。
    * **重新命名 (Rename):** 將檔案改名為 `steam_games.csv`
    * **移動檔案 (Move):** 將檔案放入 `data/raw/` 資料夾中。

2.  **抓取副資料集 (Supplementary Dataset):**
    * 前往 SteamDB 網站（分別抓取 `Most wishlisted games`、`Wishlist activity` 兩頁面）。
    * 使用 steamdb_crawler.py 收集並整理網頁資料。

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