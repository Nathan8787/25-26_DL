# HW3 · Shakespeare RNN/LSTM

## 專案概覽
- 以字元級 RNN / LSTM 建模 Shakespeare corpus（train/valid 由助教提供），並比較架構、hidden size、seq_len 對表現的影響。
- 框架：PyTorch，使用 GPU 自動啟用 AMP 混合精度；每個 batch hidden state 依作業要求重置為 0。
- 統一輸出目錄：`plots/`（學習曲線）、`outputs/`（metrics、breakpoints、生成文字）、`checkpoints/`（模型）。
- 固定隨機種子：113024510。

## 執行方式
- 安裝依賴（只需常見科學套件，未提供 requirements）：例如 `pip install torch tqdm matplotlib numpy`.
- 執行所有實驗與產生圖表、文字：
  ```bash
  python hw3.py
  ```
  - 會跑 RNN/LSTM baseline + hidden size/seq_len sweep，並存出最佳 checkpoint、CSV/JSON 指標、breakpoint 文本與所有比較圖。
  - 若 `checkpoints/best_model_LSTM.pth` 存在，程式會自動以 seed "JULIET"、temperature 0.6、長度 500 產生最終文本到 `outputs/generation_final.txt`。
- 調整超參數：修改 `Config` 以及 `build_experiments` 中的清單即可（例如 batch、seq_len、hidden_size 等）。

## 資料與檔案
- `shakespeare_train.txt`, `shakespeare_valid.txt`：訓練/驗證語料。
- `hw3.py`：主程式，跑實驗、畫圖、生成文字。
- `hw3_kaggle.py`：Kaggle 版本（若需）。
- `report.md`, `report.pdf`, `report.html`：完整報告；`Deep_Learning_2025_HW3.pdf` 為題目說明。
- `Graging_standard.txt`：評分規範備忘。

## 訓練設定（基於 report.md）
- Optimizer: Adam，lr=0.002，clip_grad_norm=5，batch=256，epochs=5（patience=5 確保每個實驗都有 breakpoints）。
- 模型：雙層 RNN/LSTM，dropout=0.25，baseline 以 hidden=128、seq_len=100；另外掃描 hidden ∈ {64, 128, 256}、seq_len ∈ {50, 100, 150}。
- 指標：loss、BPC (= loss/ln2)、accuracy/error；生成用 temperature=0.6。

## 結果摘要
- Baseline 對照（seq_len=100, hidden=128）：
  - RNN：最佳在第 1 epoch，valid loss=1.6089、BPC=2.3211、acc=0.5210。
  - LSTM：最佳在第 3 epoch，valid loss=1.4423、BPC=2.0808、acc=0.5698（較 RNN 降 0.17 loss、acc 提升約 4.9%）。
- Hidden size 掃描：
  - RNN：hs64 valid loss=1.7561（e5） < hs128=1.6089（e1） < **hs256=1.4975（e3）**。
  - LSTM：hs64 valid loss=1.5718（e5），hs128=1.4423（e3），**hs256=1.3905（e1）**，首 epoch 即達最佳。
- Seq_len 掃描：
  - RNN：seq50 valid loss=1.6221（e5），seq100=1.6089（e1），**seq150=1.5956（e3）**。
  - LSTM：seq50 valid loss=1.4948（e5），**seq100=1.4423（e3）**，seq150=1.4443（e5），表現相近但 seq100 較經濟。
- Generation 品質：LSTM 斷點文本較流暢、碎句少；hs256 明顯提升可讀性。最終 prime (JULIET, T=0.6) 生成 12 行對話格式，見 `outputs/generation_final.txt`。
- 最佳模型：`LSTM_HS_256`（seq_len=100），valid loss=1.3905、BPC≈2.006、acc≈0.587（首 epoch）。

## 可能的延伸
- 若要追求更低 loss：可延長 epoch、嘗試學習率 decay 或增加 hidden/layer 數，但目前在 5 epoch 內已達收斂。
- 調整 temperature 以控制生成多樣性；當前 0.6 已兼顧穩定度與可讀性。
