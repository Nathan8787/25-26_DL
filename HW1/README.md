# HW1 深度學習作業總覽

這是 2025 深度學習課程 HW1 專案。本REPO提供 **能源效率回歸** 與 **Ionosphere 二元分類** 兩個子題的最新、可重現實作，涵蓋：

- 純 `numpy`/`pandas`/`matplotlib`/`tqdm` 實作的前饋式神經網路與訓練流程。  
- 梯度數值檢查、特徵重要度、特徵子集合重訓 (Regression)。  
- Latent 維度比較與視覺化分析 (Classification)。  
- 對應的規格書 (`specs/`) 與專業報告 (`reports/`)。  

---

## 1. 目錄結構 Repository Layout

```
HW1/
├── datasets/                  # Energy efficiency & Ionosphere CSV
├── guidelines/                # 作業說明 PDF
├── project/
│   ├── src/                   # regression_pipeline.py, classification_pipeline.py
│   ├── artifacts/             # 模型、訓練歷史、特徵/latent 分析產物
│   ├── data/processed/        # 預留的處理後資料 (classification 子題已輸出 .npy)
│   ├── figures/               # 學習曲線、散點圖、混淆矩陣、latent 視覺化
│   └── results/               # regression_summary.json, classification_summary.json
├── reports/                   # regression_report.md、classification_report.md
└── specs/                     # 最新規格書 (regression_spec.md、classification_spec.md)
```

`Lectures/` 夾帶課程教材，與作業程式碼分開管理。

---

## 2. 環境需求 Environment

- Python 3.10+  
- 推薦套件：
  ```
  pip install numpy pandas matplotlib tqdm
  ```
- 若欲重新安裝環境，可用 `python -m venv .venv && .venv\Scripts\activate` 建立虛擬環境後安裝。

---

## 3. 快速開始 Quick Start

```bash
cd HW1/project

# Regression 任務：前處理 → 訓練 → 評估 → 特徵分析
python src/regression_pipeline.py

# Classification 任務：前處理 → 訓練 → 評估 → latent 分析
python src/classification_pipeline.py
```

兩支腳本皆無額外 CLI 參數，執行即自動完成完整流程並輸出所有指標、圖表與分析檔案。若要重新產生圖表或覆寫既有結果，請確保 `artifacts/` 與 `figures/` 可寫入。

---

## 4. 模型與超參數摘要 Model Overview

| 任務 | 架構 (含隱藏層) | 啟用函式 | 主要超參數 |
|------|----------------|-----------|------------|
| Regression | `[d, 8, 8, 4, 1]` | ReLU ×3 → Linear | lr=0.01, batch=32, clip=±1.0, patience=5000, seed=29 |
| Classification (Baseline) | `[34, 64, 32, 16, 1]` | ReLU ×3 → Sigmoid | lr=0.01, batch=32, clip=±1.0, patience=5000, min_delta=1e-6, seed=666666 |

- 兩者皆採 Xavier/Glorot initialization 與 mini-batch SGD。  
- 訓練前執行有限差分梯度檢查；若最大相對誤差 > 1e-3 會中止，實際訓練皆成功通過。  

如需完整細節，請參考 `specs/regression_spec.md` 與 `specs/classification_spec.md`。

---

## 5. 最新成果摘要 Key Results

| 任務 | Train | Val | Test | 其他亮點 |
|------|-------|-----|------|----------|
| Regression (Heating Load) | RMS 0.3067 | RMS 0.3760 | **RMS 0.4598** (MAE 0.3341) | Permutation importance 顯示 `roof_area`、`wall_area`、`relative_compactness` 最關鍵；完整特徵子集遠優於 Top-k。 |
| Classification (Ionosphere) | Acc 95.09% (Err 4.91%) | Acc 89.29% | **Acc 91.55% (Err 8.45%)** | Test Confusion：TP=37, FN=1, FP=5, TN=28；Latent=64 時測試錯誤率下降至 5.63%。 |

完整分析、表格與圖表請見 `reports/regression_report.md` 與 `reports/classification_report.md`。  

主要輸出圖檔位於 `project/figures/`，可直接嵌入正式報告：
- Regression：`regression_learning_curve.png`, `regression_pred_vs_actual_*.png`, `regression_feature_importance.png`, `regression_subset_performance.png` 等。  
- Classification：`classification_learning_curve.png`, `classification_confusion_matrix.png`, `classification_latent_*`, 不同 latent 對應的學習曲線等。  

---

## 6. 報告與規格 Reports & Specs

- **規格書 Specs**  
  - `specs/regression_spec.md`：包含資料流程、模型設計、特徵群組與執行指引。  
  - `specs/classification_spec.md`：記錄 latent 分析流程、混淆矩陣繪製等最新做法。  

- **報告 Reports**  
  - `reports/regression_report.md`：符合作業要求 (網路架構、學習曲線、Train/Test RMS、預測圖、特徵分析)。  
  - `reports/classification_report.md`：涵蓋學習曲線、錯誤率、混淆矩陣、latent 比較與視覺化。  

在準備最終繳交 PDF 時，可直接引用上述 Markdown 內容或將其轉為 PDF。  

---

## 7. 重現流程 Reproducibility Checklist

1. 確認 `datasets/` 內 CSV 檔案齊全。  
2. (可選) 清空既有輸出：`rm -rf project/artifacts project/figures project/results`。  
3. 於 `HW1/project` 依序執行兩個 pipeline (或僅執行需要的任務)。  
4. 在 `project/results/` 取得新的 `*_summary.json`；圖表與 CSV 會同步更新。  
5. 若要更新規格或報告，請重新閱讀 `specs/` 與 `reports/`，確保內容與最新數據一致。  

