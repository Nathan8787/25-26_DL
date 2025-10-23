# HW1 Task 2 Classification Specification / HW1 第二題分類任務規格書

- 版本 Version: 2.0  
- 更新日期 Last Updated: 2025-10-15  
- 撰寫者 Author: Codex Specification Assistant  

---

## 1. 任務概要 Mission Overview
- **目標 Goal**：以 Ionosphere 資料集 (351 筆樣本、34 個連續特徵) 手刻前饋式神經網路，進行 `g` (good) vs `b` (bad) 的二元分類，損失函式為交叉熵 (Binary Cross-Entropy)。  
- **限制 Constraints**：不得使用任何封裝之深度學習框架；僅依賴 `numpy`, `pandas`, `matplotlib`, `tqdm`。  
- **流程 Pipeline**：`preprocess_data → train_model → evaluate_model → run_latent_analysis` 由 `run_pipeline()` 自動串接。  
- **附加要求 Extras**：必須比較不同 latent 維度的表現，並對訓練中數個時間點的 latent representation 進行視覺化。  

---

## 2. 資料來源與欄位定義 Dataset & Features
- **資料檔案 Dataset Path**：`HW1/datasets/2025_ionosphere_data.csv`。腳本從 `project/src/classification_pipeline.py` 透過 `Path("../datasets/...")` 載入。  
- **筆數 & 切分 Samples & Split**：  
  - 總數 351。  
  - 以固定亂數種子 (seed=666666) 打散後切分：`train 224 / val 56 / test 71` (對應 64% / 16% / 20%)。  
- **欄位 Columns**：原始檔無欄名，載入後命名為 `f0 ... f33` 與 `label`。  
  - 所有 `f*` 視為 float 特徵，並以 train 均值 / 標準差做 z-score。  
  - `label`: `'g' → 1.0`, `'b' → 0.0`。  

---

## 3. 程式架構與 CONFIG Structure
- **主檔案 Main Script**：`HW1/project/src/classification_pipeline.py`。  
- **程式模組 Modules**：
  1. `CONFIG`：亂數種子、路徑、訓練超參數、latent 候選列表。  
  2. `TrainingRecord` dataclass：紀錄 epoch、train/val loss、accuracy、error rate。  
  3. 前處理：`load_dataset`, `split_dataset`, `standardize_features`。  
  4. 迷你批次產生器：`build_batches`。  
  5. 模型類別：`NeuralNetworkClassifier` (前饋網路 with ReLU + Sigmoid + latent getter)。  
  6. 損失與評估：`binary_cross_entropy`, `compute_metrics`, `gradient_check`。  
  7. 視覺化：`plot_learning_curves`, `plot_confusion_matrix`, `plot_latent_distribution`, `pca_reduce`。  
  8. 訓練：`train_model` (含梯度檢查、裁剪、早停、latent 快照)。  
  9. 評估：`evaluate_model` (保存歷史、圖表、混淆矩陣)。  
  10. Latent 比較：`run_latent_analysis`。  
  11. 入口：`run_pipeline()`。

- **CONFIG 內容**：
  - `seed = 666666`：設定 `random` 與 `numpy`。  
  - `paths`：  
    ```
    dataset = ../datasets/2025_ionosphere_data.csv
    processed_dir = data/processed/classification
    artifacts_dir = artifacts/classification
    figures_dir = figures
    results_dir = results
    best_model = artifacts/classification/best_model.pkl
    training_history = artifacts/classification/training_history.csv
    latent_dir = artifacts/classification/latents
    latent_comparison = artifacts/classification/latent_comparison.csv
    summary = results/classification_summary.json
    ```  
  - `hyperparameters`：
    | 參數 | 值 | 說明 |
    |------|----|------|
    | `learning_rate` | 0.01 | SGD 步長 |
    | `epochs` | 30000 | 訓練上限 |
    | `mini_batch_size` | 32 | 小批次大小 |
    | `gradient_clip_value` | 1.0 | 梯度裁剪範圍 [-1, 1] |
    | `early_stopping_patience` | 5000 | 連續無改善時終止 |
    | `early_stopping_min_delta` | 1e-6 | 驗證 loss 改善需超過此閾值 |
  - `latent_layer_sizes = [4, 16, 32, 64, 128]`：供 baseline 與分析迭代使用。  

---

## 4. 前處理流程 Data Preprocessing
1. **目錄建立**：`ensure_directories()` 確保 `processed_dir`, `artifacts`, `figures`, `results`, `latent_dir` 存在。  
2. **資料讀取**：`load_dataset(path)` → pandas 讀取、補欄名。  
3. **資料切分**：`split_dataset(df, seed)`  
   - 以 `sample(frac=1.0, random_state=seed)` 打散。  
   - `train_val = floor(0.8 * N)`、`test = N - train_val`。  
   - `val = floor(0.2 * train_val)`、`train = train_val - val`。  
4. **標準化與標籤轉換**：`standardize_features(splits)`  
   - 使用 train 計算均值 / 標準差 (`std==0` 時設為 1)。  
   - 對 train/val/test 套用同一組 z-score，並轉為 `np.float64`。  
   - label 映射為 `0/1` 並 reshape 成 `(N,1)`。  
   - 回傳 `(features, targets, metadata)`，其中 metadata 包含 `feature_names`, `mean`, `std`, `label_map`。  

> 註：目前流程將處理後資料保留於記憶體，未另外輸出 `.npy`；若需持久化可於 `processed_dir` 手動新增保存步驟。  

---

## 5. 模型設計 Model Architecture
- **類別 Class**：`NeuralNetworkClassifier`。  
- **結構 Layer Sizes**：`[input_dim, 64, 32, latent_size, 1]`。  
  - 隱藏層 1 (`64`)、隱藏層 2 (`32`) 使用 ReLU。  
  - 倒數第二層 (latent) 大小可變，由 `latent_size` 決定；ReLU 啟用。  
  - 輸出層 1 節點，Sigmoid activation → 機率。  
- **初始化 Initialization**：Glorot/Xavier Uniform。  
- **Forward 行為**：  
  - 儲存 activations (`A^l`)、pre-activations (`Z^l`) 於 `_cache`。  
  - 同時記錄倒數第二層輸出於 `_latent_output`，供 `capture_latent()` 及視覺化使用。  
- **Backward**：  
  - 以 `(y_pred - y_true) / N` 作為輸出層梯度 (Sigmoid + BCE)。  
  - 隱藏層乘上 ReLU mask。  
- **介面**：`predict_proba` (機率)、`predict` (0/1)、`get_latent_output` (最新 latent)、`get_parameters`/`set_parameters`。  

---

## 6. 訓練流程 Training Loop
1. **梯度檢查 Gradient Check**  
   - 預設啟用 (`perform_gradient_check=True`)，在訓練開始前執行一次。  
   - `epsilon = 1e-5`, `sample_size = 25`；若最大相對誤差 > 1e-3 則 raise。  
2. **迷你批次**：`build_batches(train_X, train_y, batch_size=32, seed=CONFIG["seed"] + epoch)`。  
3. **優化**：純 SGD，逐批前向 → 反向 → 裁剪 → 更新。  
   ```
   preds = model.forward(X_batch, training=True)
   grad_w, grad_b = model.backward(y_batch, preds)
   grad_w = clip(grad_w, -1.0, 1.0); grad_b 同理
   model.apply_gradients(..., lr=0.01)
   ```  
4. **歷史記錄**：每 epoch 以完整 train/val 資料計算 loss/accuracy/error rate，存入 `TrainingRecord`。  
5. **早停 Early Stopping**：  
   - 監控驗證 loss。  
   - 若 `best_val_loss - current_val_loss > min_delta (1e-6)` 視為改善並保存 `best_state`。  
   - 否則 `epochs_since_best += 1`；連續 5000 epoch 無改善即中止。  
6. **Latent 快照** (`store_latent=True`)：  
   - `epoch0_init`：訓練前（隨機權重）在驗證集的 latent。  
   - `epoch1_after_first_update`：第一個 epoch 後。  
  - `midpoint`：`ceil(epochs/2)` 時的 latent。  
   - `epoch<best>_best`：每次更新最佳驗證 loss 時覆蓋保存。  
   - 快照以 `.npy` 儲存於 `artifacts/classification/latents/latent_<size>[_baseline]/`，並於 `figures/` 輸出對應散點圖。  
7. **成果保存**：  
   - `history` → `artifacts/classification/training_history.csv`。  
   - `best_state` → `artifacts/classification/best_model.pkl`。  
   - `plot_learning_curves()` → `figures/classification_learning_curve.png` (單圖雙 y 軸)。  

---

## 7. 評估 Evaluation
- `evaluate_model(model, features, targets, history, history_path, learning_curve_path)`  
  - 重新保存歷史 CSV、繪製學習曲線 (若路徑不同會覆寫)。  
  - 對 train/val/test 推論 → `compute_metrics` 回傳 loss / accuracy / error rate。  
  - 生成 `figures/classification_confusion_matrix.png` (2×2, TP/FN/FP/TN)。  
  - 將結果寫入 `results/classification_summary.json`。  

---

## 8. Latent Analysis
- `run_latent_analysis(features, targets, CONFIG)`  
  - 迭代 `latent_layer_sizes = [4, 16, 32, 64, 128]`：  
    - 依當前 latent size 建立新模型 → 以同樣超參數重訓 (梯度檢查關閉以節省時間)。  
    - 保存 `training_history_latent_<size>.csv`、`classification_learning_curve_latent_<size>.png`。  
    - 評估 train/val/test error rate，彙整至 `artifacts/classification/latent_comparison.csv`。  
    - 輸出 latent `.npy` 與對應的 PCA/散點圖 `figures/classification_latent_<size>_<stage>.png`。  
  - Baseline run (`run_pipeline`) 先以 `latent_size = 16` 訓練並保存結果，再進行上述比較。  

---

## 9. 執行指南 Execution
- **完整流程**：於 `HW1/project` 目錄執行  
  ```bash
  python src/classification_pipeline.py
  ```  
  腳本會依序完成前處理 → baseline 訓練 → 評估 → latent 分析。無 CLI 參數。  
- **輸出清單 Outputs**：  
  - `results/classification_summary.json`  
  - `artifacts/classification/best_model.pkl`  
  - `artifacts/classification/training_history*.csv`  
  - `artifacts/classification/latent_comparison.csv`  
  - `artifacts/classification/latents/` 內各種 `.npy` 快照  
  - `figures/classification_learning_curve*.png`, `classification_confusion_matrix.png`, `classification_latent_*`  

---

## 10. 報告撰寫焦點 Reporting Checklist
1. **網路架構**：列出 `[34 → 64 → 32 → latent → 1]`，並說明 activation。  
2. **訓練設定**：學習率、批次大小、總 epoch、梯度裁剪、早停條件、梯度檢查結果。  
3. **學習曲線**：引用 `classification_learning_curve.png` (Loss & Accuracy) 說明收斂情形與過擬合觀察。  
4. **最終指標**：Train/Val/Test loss、accuracy、error rate。  
5. **混淆矩陣**：解讀 TP/FP/FN/TN，佐以誤判情形分析。  
6. **Latent 分析**：  
   - 條列 `latent_comparison.csv` 的各尺寸 error rate。  
   - 說明最佳 latent 大小 (例如測試錯誤率最低者) 及原因推論。  
   - 提供不同時間點的 latent scatter 圖 (至少 baseline 的 `epoch0_init`, `epoch_mid`, `epoch_best`)。  

---

## 11. 測試與驗證 Testing & QA
- **梯度檢查**：確認最大相對誤差小於 1e-3。  
- **資料切分**：驗證輸出樣本數是否為 224/56/71。  
- **流程完成度**：觀察外層 `tqdm` 顯示四個階段皆完成。  
- **再現性**：重跑腳本應得到一致的 summary、latent_comparison 結果 (差異僅浮點誤差)。  

---

## 12. 變更紀錄 Change Log
- **v2.0 (2025-10-15)**：  
  - 更新模型結構至 `[input_dim, 64, 32, latent, 1]`，並記錄 baseline latent = 16。  
  - 對齊現行程式：移除舊版 `--phase` CLI 說明，補充 `latent_dir`、`latent_comparison`、快照命名。  
  - 明確記載梯度裁剪、早停條件、梯度檢查參數。  
  - 調整報告與測試清單，對應作業指南需求 (學習曲線、錯誤率、latent 視覺化)。  

---

本規格檔已與 `classification_pipeline.py` 最新版同步。若未來調整超參數或輸出位置，請同步修改此文件以維持一致性。  
