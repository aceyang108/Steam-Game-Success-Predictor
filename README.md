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
    1.  **願望清單排名 (Wishlist Rank):** 遊戲發售前的受關注程度。
    2.  **行事曆 (Calendar):** 發售日是否撞期 Steam 大型特賣會 (Summer/Winter Sale)。
* **技術挑戰：** 由於 SteamDB 具有嚴格的反爬蟲機制 (Cloudflare)，我們開發了專用的爬蟲 (`steamdb_crawler.py`) 來獲取這些珍貴數據。

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
    * 前往下載：
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
    │       ├── wishlists_top1000.csv    
    │       ├── training_data_main.csv   
    │       └── training_data_full.csv
    ├── src/
    │   ├── Train_for_nn.py
    │   ├── Train_for_RFV2.py
    │   ├── Train_for_RF.py
    │   ├── exolain_shap.py
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
python src/preprocess.py

```

輸出：

```
data/processed/training_data_main.csv #just main

data/processed/training_data_full.csv #main + Supplementary

```

---

## 5. 訓練模型 (Training)

XGBOOST:

```
python src/train.py

```

Random Forest:

```
python src/train_for_RF.py

python src/train_for_RFV2.py

```

PyTorch Neural Network:

```
python src/train_for_nn.py

```

SHAP Analysis
```
python src/explain_shap.py

```

## 6. 專案時程 (3-Week Roadmap)

### 📅 Week 1: 基礎建設與資料清洗 (Baseline)
- [X] **Data:** 下載 Kaggle 資料集，並建立「特賣會日期表 (Supplementary)」。
- [X] **Data:** 調整preprocessing.py到配合Kaggle 資料集。
- [X] **Model:** 定義 Y 的分類門檻，跑通第一個 XGBoost 模型，取得 Baseline 準確率。

### [ ] Week 2: 模型優化與競技 (Optimization & Competition)
- [X] **Data**: 驗證加入 SteamDB 副資料 (training_data_full.csv) 是否提升準確率。
- [ ] **Model**: 對 XGBoost 與 Random Forest 進行超參數調優 (Finetune)。
- [ ] **Model**: 優化 PyTorch 網路結構 (調整層數、Dropout)。

### [ ] Week 3: 應用展示與深度分析 (Demo & Final Polish)
- [ ] **App**: 開發 Streamlit 互動網頁，展示預測結果。
- [ ] **XAI**: 實作 SHAP Analysis，解釋模型決策原因 (最後階段執行)。
- [ ] **Report**: 撰寫期末報告，比較三種模型的優劣與適用場景。