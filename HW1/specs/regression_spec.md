# HW1 Task 1 Regression Specification / HW1 第一題回歸任務規格書

- 版本 Version: 2.0  
- 更新日期 Last Updated: 2025-10-15  
- 撰寫者 Author: Codex Specification Assistant  

---

## 1. 專案概要 Project Overview
- **任務 Mission**：以能源效率資料集 (Energy Efficiency Dataset, 768 筆樣本) 手刻前饋式神經網路 (Fully Connected Network)，透過反向傳播與小批次隨機梯度下降預測建築物的 `Heating Load`。  
- **技術限制 Constraints**：全程僅可使用 `numpy`, `pandas`, `matplotlib`, `tqdm` 等基礎科學運算套件；嚴禁任何已封裝的深度學習層或優化器 (e.g. TensorFlow、PyTorch、scikit-learn NN)。  
- **核心流程 Pipeline**：`preprocess_data → train_model → evaluate_model → run_feature_selection`，由 `run_pipeline()` 自動串接並附帶進度條。  
- **其他要求 Additional Requirements**：需實作梯度數值檢查、Permutation Importance、特徵子集合重訓，並產生學習曲線與預測對照圖供報告使用。  

---

## 2. 資料來源 Data Source & Schema
- **資料檔案 Dataset Path**：`HW1/datasets/2025_energy_efficiency_data.csv` (由 `project/src/regression_pipeline.py` 透過 `Path("../datasets/...")` 讀取)。  
- **資料筆數 Samples**：768。最終切分為 `train 461 / val 115 / test 192`。  
- **欄位定義 Columns**：載入時即轉為 `snake_case`。  

| Raw Column Name                 | Internal Field              | Type     | 說明 /處理備註 |
|---------------------------------|-----------------------------|----------|----------------|
| `# Relative Compactness`        | `relative_compactness`      | float64  | 連續特徵，訓練均值 / 標準差做 z-score |
| `Surface Area`                  | `surface_area`              | float64  | 連續特徵，z-score |
| `Wall Area`                     | `wall_area`                 | float64  | 連續特徵，z-score |
| `Roof Area`                     | `roof_area`                 | float64  | 連續特徵，z-score |
| `Overall Height`                | `overall_height`            | float64  | 連續特徵，z-score |
| `Orientation`                   | `orientation`               | int      | 類別特徵 `{2,3,4,5}` → one-hot 四維 |
| `Glazing Area`                  | `glazing_area`              | float64  | 連續特徵，z-score |
| `Glazing Area Distribution`     | `glazing_area_distribution` | int      | 類別特徵 `{0..5}` → one-hot 六維 |
| `Heating Load`                  | `heating_load`              | float64  | 迴歸目標，維持原始尺度 |
| `Cooling Load`                  | `cooling_load`              | float64  | 讀入後不使用，於特徵管線中被忽略 |

---

## 3. 程式結構與設定檔 Program Structure & CONFIG
- **檔案位置**：`HW1/project/src/regression_pipeline.py`。  
- **主要區塊**：
  1. `CONFIG`：集中管理亂數種子、路徑、超參數、特徵名稱與類別對應。  
  2. `TrainingRecord` dataclass：封裝單一 epoch 的 train/val MSE、RMS。  
  3. 前處理相關函式：`load_raw_dataset`, `shuffle_and_split`, `encode_and_standardize`。  
  4. 批次生成器：`build_mini_batches`。  
  5. 模型類別：`NeuralNetwork` (前饋式全連接網路，手刻 forward/backward)。  
  6. 訓練流程：`train_model` (含梯度檢查、梯度裁剪、早停)。  
  7. 評估與視覺化：`evaluate_model`, `plot_learning_curve`, `plot_prediction_scatter`, `plot_pred_vs_label`。  
  8. 特徵分析：`permutation_importance`, `evaluate_feature_subsets`, `run_feature_selection`。  
  9. 進出檔工具：`save_training_history`, `save_json`, `load_best_model`。  
  10. 入口點：`run_pipeline()`，統一執行整個流程。

- **CONFIG 關鍵內容**：
  - `seed = 29`：同時設定 `random` 與 `numpy`。  
  - `paths`：相對於 `project/` 目錄的輸入輸出路徑 (processed/artifacts/figures/results)。  
  - `hyperparameters`：
    | 參數 | 值 | 說明 |
    |------|----|------|
    | `learning_rate` | 0.01 | 純手刻 SGD 步長 |
    | `epochs` | 30000 | 設定上限，實務由早停提前結束 |
    | `mini_batch_size` | 32 | 建築資料規模足以使用小批次 |
    | `gradient_clip_value` | 1.0 | element-wise clipping，抑制梯度爆炸 |
    | `early_stopping_patience` | 5000 | 連續無改善達上限即停止 |
  - `continuous_features`：六個連續欄位。  
  - `orientation_categories` / `orientation_feature_names`：四維 one-hot。  
  - `gad_categories` / `gad_feature_names`：六維 one-hot。

---

## 4. 資料處理流程 Data Processing
1. **目錄建立**：`ensure_directories()` 確保 `data/processed`, `artifacts/regression`, `figures`, `results` 存在。  
2. **資料讀取**：`load_raw_dataset()` 以 `pandas.read_csv(dtype=float)` 載入並重新命名欄位。  
3. **資料切分**：`shuffle_and_split()`  
   - 先以 `sample(frac=1.0, random_state=seed)` 打散。  
   - `train_full = floor(0.75 * N)`、`test = N - train_full`。  
   - `val = floor(0.2 * train_full)`、`train = train_full - val`。  
   - 重置索引後回傳字典 `{"train", "val", "test"}`。  
4. **特徵工程**：`encode_and_standardize()`  
   - 僅使用 train split 估計連續欄位的均值 / 標準差。  
   - 對 train/val/test 套用相同的 z-score。  
   - `orientation` 與 `glazing_area_distribution` 轉為 one-hot；若遇未知類別直接丟出錯誤。  
   - 合併順序：連續標準化特徵 → orientation one-hot → gad one-hot。  
   - 目標值 `y` 取 `heating_load` (shape `(N,1)`)。  
   - 回傳 `features`, `targets`, `metadata` (含均值、標準差、欄位名稱、類別對應列表)。  

---

## 5. 模型設計 Model Architecture
- **類別**：`NeuralNetwork`  
  - **層數 Layer Sizes**：`[input_dim, 8, 8, 4, 1]`。  
    - 三層隱藏層皆為 ReLU。  
    - 輸出層為線性，適用回歸。  
  - **初始化**：Glorot/Xavier Uniform，`limit = sqrt(6 / (fan_in + fan_out))`。  
  - **Forward**：依序執行仿射 + ReLU，並快取 activations (`A`) 與 pre-activations (`Z`) 供 backward 使用。  
  - **Backward**：手刻鏈式法則，MSE 導數 `dA = 2(y_pred - y_true)/m`，隱藏層乘上 ReLU mask。  
  - **Parameter I/O**：`get_parameters()` / `set_parameters()` 提供複製與還原最佳模型的能力。  

---

## 6. 訓練流程 Training Loop
1. **梯度檢查 Gradient Check**  
   - `gradient_check()` 預設啟用，`epsilon = 1e-5`，`sample_size = 10`。  
   - 對權重抽樣若干元素執行中央差分；偏置僅檢查各層第一個元素。  
   - 回傳最大相對誤差；若 >1e-3 則中止。  
2. **批次生成 Mini-batch**：`build_mini_batches(X, y, batch_size=32, seed=CONFIG["seed"] + epoch)`。  
3. **優化 Optimizer**：純 SGD，無動量。每步驟執行：  
   ```
   grad_w, grad_b = model.backward(y_batch, preds)
   grad_w = clip(grad_w, -1.0, 1.0); grad_b 同理
   model.apply_gradients(grad_w, grad_b, learning_rate=0.01)
   ```  
4. **記錄 Metrics**：每個 epoch 以全量 train、val 資料計算 MSE/RMS/MAE (透過 `compute_metrics`)。  
5. **早停 Early Stopping**：  
   - 以 validation RMS 為基準。  
   - 若 `val_rms` 嚴格下降即更新 `best_state`; 否則 `epochs_since_best += 1`。  
   - 連續 5000 個 epoch 無改善 → break。  
6. **結果保存**：  
   - `history` (list of `TrainingRecord`) → `artifacts/regression/training_history.csv`。  
   - `best_state` → `artifacts/regression/best_model.pkl` (pickle of `get_parameters()`)。  
   - `plot_learning_curve()` → `figures/regression_learning_curve.png`。  

---

## 7. 評估與輸出 Evaluation & Artifacts
- **評估函式**：`evaluate_model(model, features, targets)`  
  - 對 train/val/test 全量資料做推論，回傳 dict：`{"mse","rms","mae"}`。  
  - 產出圖檔：  
    - `figures/regression_pred_vs_actual_[train|val|test].png` (散點圖 + `y=x` 參考線)  
    - `figures/regression_pred_vs_label_[train|test].png` (逐樣本折線，觀察預測偏差)  
  - 評估結果以 `save_json` 寫入 `results/regression_summary.json`。  

---

## 8. 特徵選取 Feature Selection
1. **Permutation Importance** (`permutation_importance`)  
   - 以最佳模型在驗證集計算 baseline RMS。  
   - 針對以下特徵群組逐一置換：  
     - `G1`: `relative_compactness`  
     - `G2`: `surface_area`  
     - `G3`: `wall_area`  
     - `G4`: `roof_area`  
     - `G5`: `overall_height`  
     - `G6`: `glazing_area`  
     - `G7`: `orientation_*` (四維 one-hot 視為一組)  
     - `G8`: `gad_*` (六維 one-hot 視為一組)  
   - 輸出 `artifacts/regression/feature_importance.csv` 與條形圖 `figures/regression_feature_importance.png`。  
2. **子集合重訓** (`evaluate_feature_subsets`)  
   - 最先依重要度排序群組，預設評估：`Top-1`, `Top-3`, `Top-5`, `All`。  
   - 若 `G7` 或 `G8` 排名墊底，另外評估 `All-minus-<group>`。  
   - 對每個子集合：重建特徵矩陣 → 使用相同超參數重訓 (關閉梯度檢查以節省時間) → 評估 train/val/test RMS。  
   - 保存：  
     - `artifacts/regression/feature_subset_results.csv` (subset_id, included_groups, num_features, train/val/test RMS)。  
     - 個別訓練歷史 `training_history_<subset_id>.csv`。  
     - 圖 `figures/regression_subset_performance.png` (驗證集 RMS 對照)。  
3. **淺層解讀**：報告中需根據 CSV / 圖表說明關鍵特徵與子集合表現。  

---

## 9. 執行方式 Execution
- **完整流程**：於 `HW1/project` 目錄下執行  
  ```bash
  python src/regression_pipeline.py
  ```  
  (脚本無 CLI 參數；執行即進行前處理 → 訓練 → 評估 → 特徵分析。)  
- **輸出檔案匯總**：
  - `results/regression_summary.json`  
  - `artifacts/regression/best_model.pkl`  
  - `artifacts/regression/training_history*.csv`  
  - `artifacts/regression/feature_importance.csv`  
  - `artifacts/regression/feature_subset_results.csv`  
  - `figures/regression_*.png` (學習曲線、散點、折線、特徵重要度、子集合表現)  

---

## 10. 報告撰寫重點 Reporting Checklist (對應作業要求)
1. **模型架構**：列出層數、每層神經元數、啟用函式。  
2. **訓練詳細**：學習率、批次大小、總 epoch、早停策略、梯度裁剪、梯度檢查誤差。  
3. **學習曲線**：Train / Val RMS 隨 epoch 的圖表 (引用 `regression_learning_curve.png`)。  
4. **最終指標**：`train/val/test` RMS、MSE、MAE。  
5. **預測可視化**：`pred_vs_actual` 與 `pred_vs_label` 圖解讀。  
6. **特徵分析**：Permutation Importance 排名、不同子集合的效能比較，以及原因討論。  

---

## 11. 測試與驗證 Testing & Quality Checks
- **梯度檢查**：確認最大相對誤差 < 1e-3。  
- **資料健檢**：確保切分後 `train + val + test == 768`，並符合比例 (461/115/192)。  
- **流程健檢**：觀察 `tqdm` stage 進度是否完整 (4 個階段)。  
- **再現性**：多次執行應輸出一致的數字 (隨機種子統一於 `CONFIG["seed"] = 29`)。  

---

## 12. 變更紀錄 Change Log
- **v2.0 (2025-10-15)**：  
  - 對齊最新版 `regression_pipeline.py`：更新模型架構 ([D,8,8,4,1])、超參數 (`seed=29`, `clip=1.0`, `patience=5000`)。  
  - 調整執行說明，移除舊版 `--phase` CLI 參數，強調 `run_pipeline()` 自動執行。  
  - 補充 `pred_vs_label` 圖、Permutation Importance 群組定義、子集合輸出命名。  
  - 明確列出所有輸出檔案與報告撰寫重點。

---

本規格文件已同步最新程式碼與產出。若後續流程或超參數再調整，請同步更新本檔以維持團隊的一致依循。  
