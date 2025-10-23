# HW2 Task 2 CIFAR-10 CNN Specification / HW2 題組二 CIFAR-10 卷積神經網路規格書

- 規格 Version: 3.0  
- 最近更新 Last Updated: 2025-03-18  
- 撰寫者 Author: Codex Specification Assistant  

---

## 1. 任務概述 Mission Overview
- **核心目標 Primary Goal**：建構 CIFAR-10 影像分類系統，並完成題目 2-1 至 2-5 所要求的所有分析、比較與前處理說明。
- **技術限制 Constraints**：全面採用 TensorFlow 2.x / Keras 進行實作；需自行完成前處理與資料增強，不得依賴 AutoML 或黑箱服務；務必記錄所有套件版本與訓練流程。
- **整體流程 Pipeline**：`prepare_environment → preprocess_dataset → build_model → train_and_log → evaluate_visualize → run_regularization_study → run_preprocessing_ablation → summarize_results`，由單一腳本 `task2_cifar10_pipeline.py` 的 `run_pipeline()` 串接。
- **加值需求 Extras**：撰寫前處理章節、進行 stride/filter 與 L2 正則化比較、執行資料增強消融、視覺化特徵圖與錯誤樣本、與 MNIST 任務進行對照分析。

---

## 2. 資料來源與切分 Dataset & Split
- **下載方式 Dataset Source**：`tf.keras.datasets.cifar10` 或官方網站 `https://www.cs.toronto.edu/~kriz/cifar.html`；離線備份放置 `HW2/data/cifar10/raw/`.
- **資料切分 Data Split**：
  - 官方訓練 50,000 / 測試 10,000。
  - 由訓練集抽 5,000 作驗證（stratified, seed=20250318），最終 `train 45,000 / val 5,000 / test 10,000`.
- **資料型態 Schema**：
  - 影像：`uint8` 32×32×3。
  - 標籤：`int64`，0–9 對應 `airplane … truck`。
- **類別名稱 Class Names**：儲存在 `artifacts/task2/class_names.json` 以利報告。
- **快取 Cache**：可選擇輸出成 `.npz` 或 `.tfrecord` 至 `HW2/data/cifar10/cache/`；需在報告註記。

---

## 3. 程式結構與 CONFIG Program Structure & Configuration
- **主腳本 Main Script**：`HW2/task2_cifar10_pipeline.py`，單檔含所有函式與 `if __name__ == "__main__": run_pipeline()`。
- **核心模組 Core Components**：
  1. `CONFIG`：管控路徑、種子、超參數、進度條與 CLI 設定、前處理與實驗網格；置於檔案頂部供快速調整，支援 `--config` 以 YAML/JSON 覆寫。
  2. `ensure_directories()`：建立 `artifacts/task2`, `figures/task2`, `logs/task2`, `reports/task2`.
  3. `set_seed(seed)`：同步 `os`, `random`, `numpy`, `tensorflow`.
  4. `load_and_split_dataset(config)`：下載、切分、回傳 `(train, val, test, metadata)`.
  5. `calculate_channel_stats(train_images)`：計算 per-channel mean/std，儲存於 `artifacts/task2/channel_stats.json`.
  6. `build_datasets(data, config)`：利用 `tf.data.Dataset` 配合資料增強建構 train/val/test。
  7. `build_model(config, override=None)`：建立可調整 stride/filter 的 CNN。
  8. `compile_model(model, config)`：設定 optimizer、loss、metrics（含 top-5）。
  9. `train_and_log(model, datasets, config, tag)`：執行訓練、寫入 TensorBoard/CSV、保存 checkpoint。
 10. `evaluate_and_visualize(model, datasets, history, config, tag)`：產出學習曲線、混淆矩陣、正確/錯誤樣本、特徵圖、權重分佈。
 11. `run_stride_filter_grid(config)`：完成題目 2-1 的參數網格實驗。
 12. `run_l2_regularization_study(config)`：對應題目 2-4。
 13. `run_preprocessing_ablation(config)`：支援題目 2-5 的前處理消融。
 14. `summarize_results(config)`：匯總 CSV/JSON。
- **CONFIG 內容**：
  - `seed = 20250318`.
  - `paths = {data_root, cache_dir, artifacts_dir, figures_dir, logs_dir, reports_dir, tensorboard_dir}`。
  - `base_hyperparameters = {epochs: 120, batch_size: 256, learning_rate: 2e-4, optimizer: "adamw", weight_decay: 1e-4, warmup_epochs: 5, cosine_first_decay: 20, cosine_t_mul: 2.0, cosine_m_mul: 0.9, early_stop_patience: 12}`。
  - `augmentation = {random_flip: true, random_crop: true, crop_padding: 4, random_rotation_deg: 15, random_zoom: 0.1, random_translation: 0.1, color_jitter: {"brightness":0.2,"contrast":0.2,"saturation":0.2,"hue":0.02}, cutout: {"size":8,"prob":0.3}, mixup_alpha: 0.2, use_mixup: false}`。
  - `architecture = {blocks: 4, filters: [64,128,256,512], kernels: [3,3,3,3], strides: [1,1,1,1], dropout_rate: 0.5, use_batchnorm: true}`。
  - `experiment_grids = {stride_options: [[1,1,1,1],[1,1,2,1],[2,1,1,1]], kernel_options: [[3,3,3,3],[5,3,3,3],[5,5,3,3]], l2_lambdas: [0.0,1e-5,5e-5,1e-4,5e-4,1e-3], preprocessing_variants: ["baseline","no_standardization","no_augmentation","no_cutout","no_color","no_mixup"]}`。
  - `visualization = {correct_samples: 15, incorrect_samples: 15, feature_map_blocks: [1,2,3], feature_map_channels: 16}`。
  - `progress = {use_rich: true, transient: false, refresh_rate: 0.1, show_cpu_usage: true, stages: ["prepare_environment","preprocess_dataset","build_model","train_and_log","evaluate_visualize","run_stride_filter_grid","run_l2_regularization_study","run_preprocessing_ablation","summarize_results"]}`。
  - `cli_overrides = {"mode": ["baseline","stride_filter","l2_study","preprocessing_ablation","all"], "config": "外部設定檔路徑", "device": ["cpu","gpu","auto"], "mixed_precision": [true,false]}`。

---

## 4. 前處理流程 Data Preprocessing (Requirement 2-5)
1. **正規化 Normalization**：影像除以 255 轉 `float32`。
2. **通道標準化 Channel Standardization**：利用 train set mean/std 進行 `(x - mean) / std`；在 `metadata` 中保存。
3. **資料增強 Augmentation（僅訓練集）**：
   - 幾何：水平翻轉、32→40 padding 後隨機裁切、±15° 旋轉、±10% 平移/縮放。
   - 色彩：亮度/對比/飽和度 ±0.2、色調 ±0.02。
   - 正則化：Cutout (size=8, prob=0.3)；選配 MixUp。
4. **驗證/測試集**：僅套用 normalization + standardization。
5. **tf.data Pipeline**：
   ```python
   def build_dataset(images, labels, training):
       ds = tf.data.Dataset.from_tensor_slices((images, labels))
       if training:
           ds = ds.shuffle(10000, seed=config["seed"]).map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
       else:
           ds = ds.map(apply_normalization, num_parallel_calls=tf.data.AUTOTUNE)
       return ds.batch(config["base_hyperparameters"]["batch_size"]).prefetch(tf.data.AUTOTUNE)
   ```
6. **前處理紀錄 Documentation**：於 `reports/task2/preprocessing.md` 詳述各步驟目的、參數、效益；消融結果以表格呈現。

---

## 5. 模型架構 Model Architecture
- **Baseline CNN**：
  - Block 1: `Conv2D(64, kernel=3, stride=1, padding="same") → BatchNorm → ReLU` ×2 → `MaxPool(2)`
  - Block 2: `Conv2D(128, kernel=3)` ×2 → `BatchNorm` → `ReLU` → `MaxPool(2)`
  - Block 3: `Conv2D(256, kernel=3)` ×2 → `BatchNorm` → `ReLU` → `MaxPool(2)`
  - Block 4: `Conv2D(512, kernel=3, stride=1)` → `BatchNorm` → `ReLU` → `GlobalAveragePooling2D`
  - Head: `Dense(512, activation="relu") → Dropout(0.5) → Dense(10, activation="softmax")`
- **可調內容 Adjustable Items**：
  - `filters`, `kernels`, `strides` 由 `CONFIG["architecture"]` 或實驗 override。
  - `kernel_regularizer` 於 L2 實驗設為 `tf.keras.regularizers.L2(lambda_)`。
  - 若使用 MixUp，輸出層保持 softmax，loss 需自訂以支援混合標籤。
- **初始化 Initialization**：卷積與 Dense 層使用 `HeNormal(seed)`；BatchNorm γ 初始為 1，β 為 0。
- **模型輸出**：函式 `build_model(config, override)` 回傳 `(model, conv_layers_reference)` 以供特徵圖使用。

---

## 6. 訓練設定 Training Configuration
- **Optimizer**：`tf.keras.optimizers.experimental.AdamW(learning_rate, weight_decay)`。
- **Scheduler**：Warmup → Cosine Decay Restarts：
  ```python
  lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
      initial_learning_rate=config["base_hyperparameters"]["learning_rate"],
      first_decay_steps=config["base_hyperparameters"]["cosine_first_decay"],
      t_mul=config["base_hyperparameters"]["cosine_t_mul"],
      m_mul=config["base_hyperparameters"]["cosine_m_mul"]
  )
  ```
  Warmup 在 `train_and_log` 中手動控制。
- **Loss**：`CategoricalCrossentropy(label_smoothing=0.1)`；MixUp 時改為 `CategoricalCrossentropy(reduction='none')` 並依 MixUp 係數加權。
- **Metrics**：`CategoricalAccuracy`, `TopKCategoricalAccuracy(k=5)`, `Precision`, `Recall`, `F1Score`（可使用 `tfa.metrics.F1Score`）。
- **Callbacks**：
  - `ModelCheckpoint`（最佳 `val_accuracy` 權重）。
  - `EarlyStopping`（patience=12）。
  - `TensorBoard`, `CSVLogger`, `LearningRateLogger`, `ModelSummaryPrinter`（可自訂）。
- **進度條與階段狀態 Progress & Status Bars**：
  - 使用 `rich.progress.Progress` 建立主進度條顯示 pipeline 階段，並在 `train_and_log` 內以 `rich.progress.TaskID` 或 `tqdm.auto.tqdm` 顯示 epoch 進度、剩餘時間與當前 lr。
  - 在實驗迴圈（stride/filter、L2、消融）中為每個配置新增子任務進度條，標示 `config_id`, `current/total`。
  - 在 CLI 中提供 `--no-progress` 旗標（對應 `CONFIG["progress"]["use_rich"]`）以抑制進度列，兼容非互動環境。
- **混合精度 Mixed Precision**（可選）：若啟用 `tf.keras.mixed_precision.set_global_policy("mixed_float16")`，需在報告記錄效益與注意事項。

---

## 7. Stride & Filter 實驗 Requirement 2-1
- **執行**：`run_stride_filter_grid(config)` 遍歷 `stride_options` 與 `kernel_options`。
- **資料紀錄**：
  - `logs/task2/history_stride{...}_kernel{...}.csv`
  - `reports/task2/stride_filter_results.csv`
  - `figures/task2/stride_filter_accuracy.png`
- **分析**：於報告比較不同設定對訓練/驗證/測試績效與參數量、訓練時間的影響。

---

## 8. 正確與錯誤樣本 Requirement 2-2
- **函式**：`plot_correct_incorrect_samples(model, test_ds, config)`。
- **流程**：
  1. 取得測試集預測與 softmax 分數。
  2. 選取信心最高的 15 個正確與錯誤樣本，標註 `pred`, `label`, `confidence`.
  3. 生成 3×10 網格圖。
- **輸出**：`figures/task2/correct_vs_incorrect.png`，並將詳細資料寫入 `artifacts/task2/sample_analysis.json`。
- **報告**：討論錯誤原因（背景、姿態、顏色等）。

---

## 9. 特徵圖觀察 Requirement 2-3
- **函式**：`visualize_feature_maps(model, samples, config)`。
- **步驟**：
  1. 中介模型輸出第一、二、三卷積區塊特徵圖。
  2. 選擇 10 個代表樣本（每類 1 張），各層顯示前 16 個通道。
  3. 產生 `figures/task2/feature_maps_block{1,2,3}.png`。
- **對照說明**：於 `reports/task2/feature_map_observations.md` 比較與 MNIST 特徵圖差異。

---

## 10. L2 正則化 Requirement 2-4
- **函式**：`run_l2_regularization_study(config)`。
- **步驟**：
  1. 對每個 λ 值（含 0）建立模型並訓練。
  2. 產出 `reports/task2/l2_results.csv`，包含 train/val/test accuracy、val loss、best epoch、weight norm。
  3. 生成權重分佈圖 `figures/task2/l2_weights_lambda_{value}.png`。
- **報告**：討論最佳 λ、過擬合抑制效果、與 MNIST L2 結果之異同。

---

## 11. 前處理消融 Requirement 2-5
- **函式**：`run_preprocessing_ablation(config)`。
- **消融設定**：`baseline`, `no_standardization`, `no_augmentation`, `no_cutout`, `no_color`, `no_mixup`（若 baseline 未啟用 MixUp 則該設定可跳過或與 `augmentation.use_mixup` 連動）。
- **流程**：針對每個設定執行訓練、記錄性能、產生對應學習曲線。
- **輸出**：
  - `reports/task2/preprocessing_ablation.csv`
  - `figures/task2/preprocessing_ablation.png`（驗證集 accuracy 對照圖）
  - `reports/task2/preprocessing.md`（文字說明：理由、設定、結果、結論）
- **關聯**：於報告中強調前處理對 CIFAR-10 的必要性，並與 MNIST 任務對照。

---

## 12. 評估與指標 Evaluation & Metrics
- **最終指標**：`accuracy`, `top5_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `log_loss`.
- **學習曲線**：`figures/task2/learning_curve_baseline.png`。
- **混淆矩陣**：`figures/task2/confusion_matrix.png`（十類 10×10）。
- **分類報告**：`reports/task2/classification_report.csv`。
- **Performance Baseline**：測試集 Top-1 accuracy ≥ 0.86；優化目標 ≥ 0.90，Top-5 ≥ 0.98。
- **整體摘要**：`reports/task2/summary.json`，包含最佳設定、指標、訓練時間、觀察筆記。

---

## 13. 執行流程 Execution
- **指令**：
  ```bash
  cd HW2
  python task2_cifar10_pipeline.py --mode baseline
  python task2_cifar10_pipeline.py --mode stride_filter
  python task2_cifar10_pipeline.py --mode l2_study
  python task2_cifar10_pipeline.py --mode preprocessing_ablation
  ```
  `--mode all` 會依序執行全部步驟。`--config` 可選擇自訂 YAML（如需外部設定）。
- **狀態欄展示**：執行腳本時應看到 `rich` 進度面板，列出所有 stages、當前任務、完成百分比與耗時；在 `--mode all` 時會動態顯示每次訓練的子進度條與總體累積進度。
- **輸出檢查**：
  - `artifacts/task2/best_model_*.h5`
  - `logs/task2/history_*.csv`
  - `figures/task2/*.png`
  - `reports/task2/*.csv`, `summary.json`, `preprocessing.md`, `feature_map_observations.md`

---

## 14. 測試與品質保證 Testing & QA
- **Sanity Checks**：
  - `run_sanity_checks()` 驗證資料 shape、標準化均值/方差是否為 0/1。
  - 確認模型輸出 shape `(batch, 10)`，top-1 + top-5 指標可運作。
  - 單步梯度更新測試 loss 應下降。
- **單元測試**（可整合於腳本或獨立檔案）：
  - `test_augmentations_preserve_shape`: 確保增強後仍為 32×32×3。
  - `test_preprocessing_variants`: 各消融模式應產生合理的 datasource。
  - `test_model_build_override`: 驗證 override stride/kernel 時模型仍可編譯。
- **環境紀錄**：保存 `pip freeze` 至 `reports/task2/pip_freeze.txt`; 記錄 GPU 型號、CUDA 版本。

---

## 15. 報告撰寫指引 Reporting Checklist
1. 描述資料前處理與增強策略、提供表格與圖示。
2. 完整列出模型結構、參數數量、訓練設定。
3. 呈現 stride/filter 實驗結果與討論最佳組合。
4. 展示學習曲線、混淆矩陣、分類報告、正確/錯誤樣本。
5. 描述特徵圖觀察與與 MNIST 的比較。
6. 分析 L2 正則化趨勢、最佳 λ 以及對泛化的影響。
7. 呈現前處理消融結果，說明每一項前處理的貢獻。
8. 總結整體性能、限制與改進方向。

---

## 16. 變更紀錄 Change Log
- **v3.0 (2025-03-18)**：重寫為單腳本規格並補強前處理、資料增強、實驗設計、評估與報告細節，配合題目需求提供雙語說明。

---

本規格作為 HW2 題組二的實作準則，若未來調整流程或引入新技術，請同步更新此文件並維持版次紀錄。
