
<div align="center">
  <span style="color:#111827; font-size:40px; font-weight:700;">
    HW1 Classification Report
  </span><br>
  <span style="color:#111827; font-size:40px; font-weight:600;">
    113024510 廖振宇
  </span>
</div>

## 1. Executive Summary 摘要
- 以純 `numpy` 手刻的Forward Classification Neural Network (FCNN) 在 第二題的資料集 `Ionosphere` 上達成 **Test Accuracy = 91.55% / Error Rate = 8.45%**。  
- Baseline 架構 `[34 → 64 → 32 → 16(Baseline) → 1]` 於 **第 99 個 epoch** 取得最佳驗證表現 (`Val Loss = 0.271`, `Val Accuracy = 89.29%`, `Val Error Rate = 10.71%`)；並且也順利通過梯度檢查 (`ε=1e-5`, 相對誤差 < 1e-3)。  
- 對第四層中的node(原先的size = 16)數量進行置換(4、16、32、64、128)並訓練後分析其模型結果，分析顯示 `latent=64` 時測試錯誤率最低 (5.63%)，且 latent 投影呈現更清晰的類別分離。  

---

## 2. Data & Preprocessing 資料與前處理
- **資料集 Dataset**：`HW1/datasets/2025_ionosphere_data.csv`，共 351 筆樣本。  
- **切分策略 Split**：依作業規範打散後切為 `Train 224 / Val 56 / Test 71` (64% / 16% / 20%)。  
- **標準化與標籤轉換**：  
  - 34 個數值特徵以訓練集均值與標準差進行 z-score，並保存於 metadata。  
  - 標籤 `'g' → 1.0`, `'b' → 0.0`；最終輸入/輸出皆為 `float64`。  
- **輸出檔案 Artifacts**：  
  - `artifacts/classification/training_history.csv`：完整訓練紀錄。  
  - `results/classification_summary.json`：Train/Val/Test loss、accuracy、error rate。  

---

## 3. Model Architecture & Hyperparameters 模型與超參數
- **網路結構 Architecture**：`[34, 64, 32, latent, 1]` 
  - latent 節點數量可調整 (4, 16, 32, 64, 128)。  
  - 隱藏層 ReLU、輸出層 Sigmoid。  
  - Baseline latent 維度 = 16 (取 `latent_layer_sizes` 第二個元素)。  
  - 權重以 Xavier Uniform 初始化；`get_latent_output()` 提供倒數第二層特徵。  
- **Hyperparameters**：
  | 參數 | 值 |
  |------|----|
  | Learning Rate | 0.01 |
  | Epoch 上限 | 30,00 |
  | Mini-batch Size | 32 |
  | Gradient Clip | ±1.0 |
  | Early Stopping (`patience`, `min_delta`) | 3,00, `1e-6` (監控 Val Loss) |
  | Random Seed | 666,666 |
  | Gradient Check | `epsilon=1e-5`, 隨機抽取 25 個權重 & Bias；最大相對誤差 < 1e-3 |
- **訓練特色 Highlights**：  
  - 每個 epoch 重新打亂批次 (`seed + epoch`)。  
  - 預設儲存以下 latent 的模型狀況：`epoch0_init`, `epoch1_after_first_update`, `epoch_mid`, `epoch_best`。  
  - 透過 `TrainingRecord` 追蹤 Train/Val Loss & Accuracy；最佳權重存於 `artifacts/classification/best_model.pkl`。  

---

## 4. Learning Dynamics & Best Epoch 學習歷程
- **最佳驗證點 Best Epoch(Baseline)**：第 99 epoch  
  | Metric | Train | Val |
  |--------|-------|-----|
  | Loss   | 0.1359 | 0.2712 |
  | Accuracy | 95.09% | 89.29% |
  | Error Rate | 4.91% | 10.71% |
- **學習曲線 Learning Curves**：  
  ![Learning Curve](../project/figures/classification_learning_curve.png)  
  - Loss 自 0.68 降至 <0.3，Val Loss 於 100 epoch 左右達最低並輕微回升，可以看出在100以後繼續訓練會有點overfitting的跡象，因此 early stopping 在此發揮作用。  
  - Accuracy 曲線顯示訓練過程平滑且無劇烈震盪，表明梯度裁剪有效穩定收斂。
  - 整體而言，學習曲線反映出模型在訓練過程中表現穩定且有效，early stopping 有助於防止過擬合。  

---

## 5. Final Performance 最終效能(Baseline)
- **整體指標 Overall Metrics**：

  | Split | Loss | Accuracy | Error Rate |
  |-------|------|----------|------------|
  | Train | 0.1359 | 95.09% | 4.91% |
  | Val   | 0.2712 | 89.29% | 10.71% |
  | Test  | 0.2679 | **91.55%** | **8.45%** |

- **測試集混淆矩陣 Test Confusion Matrix**：  
  ![Confusion Matrix](../project/figures/classification_confusion_matrix.png)  
  - 數值 (由最佳權重重新推論)：
    - TP = 37, FN = 1  
    - FP = 5, TN = 28  
  - 衍生指標 Derived Metrics：Precision = 0.881、Recall = 0.974、F1 = 0.925。  
  - 觀察：模型對正類 (良好訊號 `g`) 的召回率極高；誤判主要集中於少數 false positive。

---

## 6. Latent Nodes Analysis 

### 6.1 Baseline Latent (16-d)
- 抽樣視覺化 (驗證集)：  
  | 訓練階段 Stage | 2D 投影圖 | 說明 |
  |----------------|-----------|------|
  | 初始 Initialization | ![Latent Init](../project/figures/classification_latent_baseline_epoch0_init.png) | 類別混雜，僅呈現隨機雜訊。 |
  | 第 1 次更新 After First Update | ![Latent First](../project/figures/classification_latent_baseline_epoch1_after_first_update.png) | 早期權重已將少數樣本分離。 |
  | 最佳驗證 Best | ![Latent Best](../project/figures/classification_latent_baseline_epoch99_best.png) | `g` 與 `b` 形成明顯集群，僅少數邊界點重疊。 |


### 6.1 整體 Latent Sweep Visualization
| 訓練階段 \ Latent Dim | 4 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Initialization (`epoch = 0`) | ![latent-4-e0](../project/figures/classification_latent_4_epoch0_init.png) | ![latent-16-e0](../project/figures/classification_latent_16_epoch0_init.png) | ![latent-32-e0](../project/figures/classification_latent_32_epoch0_init.png) | ![latent-64-e0](../project/figures/classification_latent_64_epoch0_init.png) | ![latent-128-e0](../project/figures/classification_latent_128_epoch0_init.png) |
| After First Update (`epoch = 1`) | ![latent-4-e1](../project/figures/classification_latent_4_epoch1_after_first_update.png) | ![latent-16-e1](../project/figures/classification_latent_16_epoch1_after_first_update.png) | ![latent-32-e1](../project/figures/classification_latent_32_epoch1_after_first_update.png) | ![latent-64-e1](../project/figures/classification_latent_64_epoch1_after_first_update.png) | ![latent-128-e1](../project/figures/classification_latent_128_epoch1_after_first_update.png) |
| Best | ![latent-4-best](../project/figures/classification_latent_4_epoch121_best.png) | ![latent-16-best](../project/figures/classification_latent_16_epoch99_best.png) | ![latent-32-best](../project/figures/classification_latent_32_epoch119_best.png) | ![latent-64-best](../project/figures/classification_latent_64_epoch101_best.png) | ![latent-128-best](../project/figures/classification_latent_128_epoch93_best.png) |

- 可以看出幾乎所有的latent維度在經過訓練後都能有效分離兩類樣本，而其中又以latent=64的效果最佳，經過訓練後可以幾乎完整的分離兩類樣本，僅有少數邊界點重疊。

### 6.2 Latent Dimension Sweep
- `artifacts/classification/latent_comparison.csv` 摘要：  

  | Latent Size | Train Err | Val Err | Test Err |
  |-------------|-----------|---------|----------|
  | 4  | 1.79% | 8.93% | 8.45% |
  | 16 | 4.91% | 10.71% | 8.45% |
  | 32 | 3.13% | 8.93% | 7.04% |
  | 64 | 4.02% | 8.93% | **5.63%** |
  | 128 | 5.36% | 8.93% | 8.45% |

- **分析 Observations**：  
  - Latent 維度 4 雖在訓練集表現優異，但測試錯誤率與 baseline 相近，顯示泛化的能力有限。  
  - 64 維達到最佳測試錯誤率 5.63%，同時維持穩定的驗證錯誤率；而從上方的圖也能看出在這個設定下的訓練後，兩類距離更遠，邊界更清晰。且64的latent所帶來的訓練/測試落差也不大，顯示其泛化能力較佳。
  - 128 維雖然有較高的訓練錯誤率，但測試錯誤率反而回升至 8.45%，暗示過高維度可能導致過擬合。

---


## 7. Appendix 附錄
- **主要檔案**：  
  - 程式：`project/src/classification_pipeline.py`  
  - 設定：`classification_pipeline.CONFIG`  
  - 最佳權重：`artifacts/classification/best_model.pkl`  
  - Latent 比較：`artifacts/classification/latent_comparison.csv`、`artifacts/classification/latents/`  
  - 圖表：`project/figures/classification_*.png`  
- **重現步驟 Reproducibility**：  
  ```bash
  cd HW1/project
  python src/classification_pipeline.py
  ```  
  將自動完成前處理、baseline 訓練、評估與 latent 分析，並重新生成所有指標與圖表。  

---
