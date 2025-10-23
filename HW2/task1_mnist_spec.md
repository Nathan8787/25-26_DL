# HW2 Task 1 MNIST CNN Specification / HW2 題組一 MNIST 卷積神經網路規格書

- 規格 Version: 3.0  
- 最近更新 Last Updated: 2025-03-18  
- 撰寫者 Author: Codex Specification Assistant  

---

## 1. 任務概述 Mission Overview
- **核心目標 Primary Goal**：以卷積神經網路（CNN）完成 MNIST 手寫數字分類，並滿足題目 1-1 至 1-4 的所有分析需求。
- **技術限制 Constraints**：採用 TensorFlow 2.x / Keras 為唯一深度學習框架，需完整記錄版本與訓練流程，禁止使用自動 AutoML 或黑箱雲端服務。
- **流程架構 Pipeline**：`prepare_environment → preprocess_dataset → build_model → train_and_log → evaluate_and_visualize → run_l2_study → summarize_results`；所有步驟由單一腳本 `task1_mnist_pipeline.py` 以 `run_pipeline()` 串接。
- **附加要求 Extras**：比較不同 stride 與 filter size、分析正確與錯誤樣本、觀察特徵圖、加入 L2 正則化並討論影響。

---

## 2. 資料來源與格式 Dataset & Format
- **下載路徑 Dataset Source**：使用 `tf.keras.datasets.mnist`；如需離線紀錄，保存於 `HW2/data/mnist/raw/`.
- **資料切分 Data Split**：保持官方分割 60,000/10,000；再由訓練集切出 5,000 作驗證（stratified, seed=20250318）。最終比例：`train 55,000 / val 5,000 / test 10,000`.
- **資料型態 Data Schema**：
  - 影像：`uint8` 28×28 灰階。
  - 標籤：`int64`，對應 0–9。
- **快取策略 Caching**：可選擇在 `HW2/data/mnist/cache/` 存 `.npz` 或 `.pt`；報告需說明是否快取。

---

## 3. 程式結構與 CONFIG Program Structure & Configuration
- **主腳本 Main Script**：`HW2/task1_mnist_pipeline.py`（單檔設計，內含所有函式與 `if __name__ == "__main__": run_pipeline()`）。
- **核心元件 Core Components**：
  1. `CONFIG` 常數：集中紀錄路徑、隨機種子、超參數、實驗網格與進度回報設定，位於檔案最上方，允許以 Python dict 形式快速調整；支援以 JSON/YAML 覆寫（例如 `--config configs/task1_override.yaml`）。
  2. `ensure_directories()`：建立 `artifacts/task1`, `figures/task1`, `logs/task1`.
  3. `set_seed(seed: int)`：同步設定 `os`, `random`, `numpy`, `tensorflow`。
  4. `load_and_split_dataset(config)`：下載、切分、正規化資料並回傳 `(train_ds, val_ds, test_ds, metadata)`.
  5. `build_datasets(train, val, test, config)`：使用 `tf.data.Dataset` 建立批次與預取。
  6. `build_model(config)`：依設定產生 Keras 模型；可切換 stride/filter。
  7. `compile_model(model, config)`：指定 optimizer、loss、metrics。
  8. `train_and_log(model, datasets, config)`：訓練、儲存 checkpoint、TensorBoard log。
  9. `evaluate_and_visualize(model, datasets, history, config)`：產出學習曲線、正確/錯誤樣本、混淆矩陣、特徵圖、權重分佈。
 10. `run_stride_filter_experiments(config)`：迭代設定網格並收集結果。
 11. `run_l2_study(config)`：覆寫 `kernel_regularizer`，比較多個 λ。
 12. `summarize_results(config)`：彙整 CSV / JSON，寫入最終摘要。
- **CONFIG 內容**：
  - `seed = 20250318`.
  - `paths = {data_root, cache_dir, artifacts_dir, figures_dir, logs_dir, tensorboard_dir, metrics_csv, summary_json}`。
  - `base_hyperparameters = {epochs: 50, batch_size: 128, learning_rate: 1e-3, optimizer: "adam", beta1: 0.9, beta2: 0.999, epsilon: 1e-7, lr_patience: 3, lr_factor: 0.5, early_stop_patience: 7}`。
  - `architecture = {conv_blocks: 3, filters: [32,64,128], kernel_sizes: [3,3,3], strides: [1,1,1], use_batchnorm: true, dropout_rate: 0.5}`。
  - `experiment_grids = {stride_options: [[1,1,1],[1,1,2],[2,1,1]], kernel_options: [[3,3,3],[5,3,3],[5,5,3]], l2_lambdas: [0.0,1e-5,1e-4,5e-4,1e-3]}`。
  - `visualization = {num_correct: 10, num_incorrect: 10, feature_map_depths: [0,1,2], feature_map_channels: 16}`。
  - `progress = {use_rich: true, transient: false, bar_refresh_rate: 0.1, stages: ["prepare_environment","preprocess_dataset","build_model","train_and_log","evaluate_and_visualize","run_stride_filter_experiments","run_l2_study","summarize_results"]}`。
  - `cli_overrides = {"mode": ["baseline","stride_filter","l2_study","all"], "config": "可選外部 YAML 路徑", "device": ["cpu","gpu","auto"]}`。

---

## 4. 資料前處理 Data Preprocessing
1. **正規化 Normalization**：影像轉 `float32`，除以 255；若啟用 Z-score 需註記並僅以訓練集計算。
2. **維度調整 Shape Handling**：`np.expand_dims(x, axis=-1)` → `(N,28,28,1)`。
3. **標籤處理 Labels**：使用 `tf.keras.utils.to_categorical(y, 10)` 產生 one-hot 向量；額外保留原始整數標籤供混淆矩陣使用。
4. **資料集建立 Dataset Creation**：
   ```python
   ds = (tf.data.Dataset.from_tensor_slices((images, labels))
           .cache()
           .shuffle(10000, seed=config["seed"])
           .batch(config["base_hyperparameters"]["batch_size"])
           .prefetch(tf.data.AUTOTUNE))
   ```
   驗證/測試集移除 shuffle。
5. **記錄 Metadata**：儲存 `mean`, `std`, `train_count`, `val_count`, `test_count`, `class_names` 至 `artifacts/task1/mnist_metadata.json`。

---

## 5. 模型架構 Model Architecture
- **Baseline CNN**：
  - `Input(shape=(28,28,1))`
  - Block 1: `Conv2D(32, kernel=3, stride=1, padding="same") → ReLU → BatchNorm → Conv2D(32, kernel=3) → ReLU → BatchNorm → MaxPool(2)`
  - Block 2: `Conv2D(64, kernel=3, stride=1) → ReLU → BatchNorm → Conv2D(64, kernel=3) → ReLU → BatchNorm → MaxPool(2)`
  - Block 3: `Conv2D(128, kernel=3, stride=1) → ReLU → BatchNorm → GlobalAveragePooling2D()`
  - Dense Head: `Dense(128, activation="relu") → Dropout(0.5) → Dense(10, activation="softmax")`
- **可調參數 Adjustable Parameters**：
  - `kernel_sizes`, `strides`, `filters` 由 `CONFIG["architecture"]` 控制。
  - `kernel_regularizer` 預設 None，於 L2 實驗覆寫為 `tf.keras.regularizers.L2(lambda_)`.
  - `dropout_rate` 可於 YAML/profile 調整。
- **初始化 Initialization**：全部卷積與全連接層採用 `HeNormal(seed=config["seed"])`。
- **建構函式**：`build_model(config, override=None)`，`override` 可指定 stride/filter/l2 以供實驗呼叫。

---

## 6. 訓練流程 Training Procedure
- **編譯 Compile**：`model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])`。
- **Callbacks**：
  - `EarlyStopping`（monitor=`val_accuracy`, patience=7, restore_best_weights=True）。
  - `ReduceLROnPlateau`（monitor=`val_loss`, factor=0.5, patience=3, min_lr=1e-5`）。
  - `ModelCheckpoint` → `artifacts/task1/best_model_stride{...}_kernel{...}.h5`
  - `TensorBoard` → `logs/task1/tensorboard/experiment_name`
  - `CSVLogger` → `logs/task1/history_{experiment}.csv`
- **訓練流程**：
  ```python
  progress.start_task("train_and_log")
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=config["base_hyperparameters"]["epochs"],
      callbacks=callback_list,
      verbose=2
  )
  progress.update("train_and_log", completed=epochs_ran)
  ```
- **訓練時間與資源**：記錄每次 fit 的 wall-clock（`time.time()`），並寫入 `artifacts/task1/training_durations.json`；若使用 GPU，額外記錄 `nvidia-smi` 結果。
- **進度條與狀態欄 Progress Reporting**：
  - 使用 `rich.progress.Progress`（或 `tqdm.rich`) 建立全域狀態欄，顯示目前階段（stage）、已完成項目、耗時與 GPU/CPU 標籤。
  - 每個核心函式進入與完成時呼叫 `progress.start_task(stage_name)` 與 `progress.update(...)`，確保 CLI 會依序顯示：
    1. `prepare_environment`
    2. `preprocess_dataset`
    3. `build_model`
    4. `train_and_log`（內含 epoch 子進度條）
    5. `evaluate_and_visualize`
    6. `run_stride_filter_experiments`
    7. `run_l2_study`
    8. `summarize_results`
  - 在實驗迴圈（stride/filter、L2）中為每個組合建立子任務進度條，以顯示完成比例與預估剩餘時間。

---

## 7. Stride & Filter 實驗 Experimentation (Requirement 1-1)
- **執行方式 Execution**：`run_stride_filter_experiments()` 迴圈遍歷 `experiment_grids["stride_options"]` 與 `["kernel_options"]`。
- **結果儲存 Outputs**：
  - `logs/task1/history_stride{...}_kernel{...}.csv`
  - `reports/task1/stride_filter_summary.csv`（包含 train/val/test accuracy、最佳 epoch、參數量）。
  - `figures/task1/stride_filter_accuracy.png`（柱狀圖比較）。
- **分析 Analysis**：於報告撰寫 stride 過大造成訊息流失、kernel 過大導致參數增加等觀察。

---

## 8. 正確與錯誤樣本分析 Requirement 1-2
- **函式 Function**：`plot_correct_incorrect_samples(model, test_ds, config)`。
- **流程**：
  1. 取得所有測試預測與分數。
  2. 選取信心最高的 10 個正確樣本與錯誤樣本（若不足則全部列出）。
  3. 以 `matplotlib` 產生 2×10 網格，標註 `pred=`, `label=`, `confidence=`.
- **輸出**：`figures/task1/correct_vs_incorrect.png`; 同時將原始索引寫入 `artifacts/task1/sample_inspection.json`。
- **報告重點**：分析常見錯誤（如 4/9、3/5 混淆）並描述潛在原因。

---

## 9. 特徵圖觀察 Requirement 1-3
- **函式**：`visualize_feature_maps(model, samples, config)`。
- **步驟**：
  1. 建立中介模型 `tf.keras.Model(model.input, [layer.output for layer in conv_layers])`.
  2. 從測試集中選取 5 個代表樣本（digits 0–4）；輸入中介模型取得 feature map。
  3. 每層取前 16 個通道，製作網格圖，並於圖說描述從邊緣偵測到部件組合的變化。
- **輸出**：`figures/task1/feature_maps_block{1,2,3}.png`.
- **文字說明**：記錄於 `reports/task1/feature_map_observations.md`。

---

## 10. L2 Regularization 研究 Requirement 1-4
- **函式**：`run_l2_study(config)`。
- **流程**：
  1. 迭代 `l2_lambda` 值；每次呼叫 `build_model(..., override={"l2": lambda})`.
  2. 重新訓練並保存最終 test accuracy、train/val loss gap。
  3. 產出權重分佈圖 `figures/task1/l2_lambda_{value}_weights.png`。
- **報告要求**：引用題目公式 `E = -1/N Σ Σ y_{nk} ln t_{nk} + α ||ω||²₂`，說明 α 與 λ 的對應；比較 λ=0 與最佳 λ 在過擬合與收斂速度上的差異。
- **結果彙整**：`reports/task1/l2_results.csv`（欄位：lambda, train_acc, val_acc, test_acc, val_loss, best_epoch, param_norm）。

---

## 11. 評估與度量 Evaluation & Metrics
- **最終評估**：於測試集計算 `accuracy`, `precision`, `recall`, `f1_macro`, `confusion_matrix`.
- **學習曲線**：`figures/task1/learning_curve_baseline.png`（顯示 loss & accuracy）。
- **混淆矩陣**：`figures/task1/confusion_matrix.png`，顯示每類預測分布。
- **摘要檔**：`reports/task1/summary.json`，記錄：
  ```
  {
    "best_config": {"strides": ..., "kernels": ..., "l2": ...},
    "metrics": {"train_acc": ..., "val_acc": ..., "test_acc": ..., "f1_macro": ...},
    "training_time_minutes": ...,
    "notes": "Observations..."
  }
  ```
- **最低標準**：測試正確率 ≥ 0.99；若未達需在報告解釋原因與補救措施。

---

## 12. 執行流程 Execution
- **指令 Command**：
  ```bash
  cd HW2
  python task1_mnist_pipeline.py --mode baseline
  python task1_mnist_pipeline.py --mode stride_filter
  python task1_mnist_pipeline.py --mode l2_study
  ```
  `--mode` 可選 `baseline`, `stride_filter`, `l2_study`, `all`；預設 `baseline`.
- **輸出檢查 Output Checklist**：
  - `artifacts/task1/best_model_*.h5`
  - `logs/task1/history_*.csv`
  - `figures/task1/*.png`
  - `reports/task1/*.csv`, `summary.json`, `feature_map_observations.md`

---

## 13. 測試與品質保證 Testing & QA
- **Unit Tests**：在同一腳本中提供 `if __name__ == "__main__":` 前的 `run_sanity_checks()` 用於快速驗證：
  - 資料形狀與 dtype。
  - 模型 forward pass 輸出維度 `(batch, 10)`。
  - 單步訓練後 loss 下降（dummy run）。
- **重現性 Reproducibility**：固定 seed；在訓練前列印 seed 與 TensorFlow 版本；保存 `pip freeze` 至 `reports/task1/pip_freeze.txt`.
- **程式風格 Code Style**：使用 `ruff` 或 `black` 作為建議工具（可在報告註記）。

---

## 14. 報告撰寫指引 Reporting Checklist
1. 描述資料來源、切分與前處理方式。
2. 詳列模型結構（層序、kernel/filter、激活函式、參數量）。
3. 說明訓練設定（optimizer、learning rate 排程、early stopping）。
4. 呈現 stride/filter 實驗結果與分析。
5. 展示學習曲線、混淆矩陣、正確/錯誤樣本並解析原因。
6. 描述特徵圖隨深度變化的觀察。
7. 討論 L2 正則化影響與最佳 λ。
8. 總結整體性能並提出可能改進方向。

---

## 15. 變更紀錄 Change Log
- **v3.0 (2025-03-18)**：依使用者要求重寫為單腳本規格，補充模型建立細節、訓練流程、實驗輸出，以及雙語報告指引。

---

此規格為 HW2 題組一的權威實作依據，若後續實作需要新增功能或調整流程，請同步更新本文件並保持版本紀錄。
