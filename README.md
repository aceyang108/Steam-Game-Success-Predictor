🎮 Steam 遊戲首週熱度預測系統 (Steam Game Success Predictor)
1. 專案簡介 (Project Overview)
本專案旨在利用機器學習模型，根據 Steam 遊戲在「發售前」的公開資訊（如價格、標籤、發行商），預測該遊戲發售後的市場表現（以首週最高在線人數 Peak CCU 為指標）。

我們希望透過此模型幫助獨立遊戲開發者了解：

哪些特徵（Feature）最能影響遊戲熱度？

選擇在什麼時間點（Timing）發售能避開競爭，獲得最大流量？

2. 資料集架構 (Dataset Architecture)
本專案嚴格遵循 "Major + Supplementary" 的雙層資料架構，以模擬真實世界的變數影響。

🔹 Major Dataset (主資料集)：遊戲本體數據
來源： Kaggle (Steam Store Games) + SteamDB

角色： 提供「內部屬性」 (Internal Factors)。

內容：

特徵 (X): 遊戲名稱、價格 (Price)、標籤 (Tags/Genres)、支援語言數、開發商歷史評價。

預測目標 (Y): 歷史最高同時在線人數 (All-time Peak CCU)。

標籤定義： 將 CCU 離散化為三類：冷門 (Cold) / 普通 (Normal) / 爆款 (Hot)。

🔸 Supplementary Dataset (副資料集)：市場時機數據
來源： Steam Sale History & Global Holidays (公開紀錄整理)

角色： 提供「外部環境影響」 (External Environmental Factors)。

類比： 如同預測 YouBike 流量需要「天氣數據」，預測遊戲銷量需要「市場檔期」。

內容：

歷年 Steam 大型特賣日期 (夏特、冬特)。

主要市場 (US/EU) 國定假日。

整合方式： 透過 Release Date 與主資料集串接，生成 is_sale_period (是否撞期特賣)、is_holiday_season (是否為旺季) 等特徵。

3. 技術堆疊 (Tech Stack)
我們選擇適合表格數據 (Tabular Data) 且具備高可解釋性的技術方案。

語言： Python

核心模型： XGBoost / Random Forest (處理結構化數據表現最佳)

資料處理： Pandas, NumPy

可解釋性 AI (XAI)： SHAP (用來解釋為什麼模型認為某款遊戲會紅)

互動展示： Streamlit (建構 Web App 供使用者動態輸入參數並查看預測結果)

4. 專案時程 (3-Week Roadmap)
📅 Week 1: 基礎建設與資料清洗 (Baseline)
[ ] Data: 下載 Kaggle 資料集，並建立「特賣會日期表 (Supplementary)」。

[ ] Data: 清洗資料，處理缺失值，將 Tags 轉換為 One-hot Encoding。

[ ] Model: 定義 Y 的分類門檻，跑通第一個 XGBoost 模型，取得 Baseline 準確率。

📅 Week 2: 模型優化與可解釋性 (Optimization & XAI)
[ ] Feature: 將副資料集特徵 (is_sale_period) 加入訓練，比較準確率是否提升。

[ ] XAI: 實作 SHAP Analysis，產出特徵重要性圖表 (Feature Importance Plot)。

[ ] Analysis: 分析並記錄有趣發現 (例如：Indie 遊戲是否特別害怕撞期夏特？)。

📅 Week 3: 互動展示與報告 (Demo & Presentation)
[ ] Demo: 使用 Streamlit 製作預測介面 (輸入遊戲資訊 -> 輸出預測等級)。

[ ] Report: 撰寫期末報告，強調「資料來源的邏輯 (Major vs Supplementary)」與「模型解釋 (SHAP)」。

[ ] Final: 準備 PPT 與 Live Demo 腳本。
