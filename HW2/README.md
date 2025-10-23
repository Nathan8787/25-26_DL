# HW2 深度學習作業說明

本專案實作 2025 年秋季深度學習課程作業（二），包含：

- **Task 1 – MNIST**：卷積神經網路之 stride/filter 掃描、L2 正則化實驗、特徵圖與樣本分析。  
- **Task 2 – CIFAR-10**：自訂資料前處理與增強，延伸 MNIST 的所有實驗並新增前處理消融。

最新成果彙整於 `task1_report.md`、`task2_report.md`，建議先閱讀後再依指引重現實驗。

---

## 1. 環境準備

本專案建議使用 Conda 建立 Python 3.12 環境並啟用 GPU 版 TensorFlow。

```bash
conda create -n DL python=3.12
conda activate DL
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU 注意事項**：TensorFlow 2.20 需搭配 CUDA 12.3 / cuDNN 9.3。若於 WSL 執行，請先確認 `nvidia-smi` 可用，再於環境中安裝 `cuda-toolkit` 與 `cudnn` 之 Conda 套件。

---

## 2. 目錄結構與重要輸出

| 路徑 | 說明 |
|------|------|
| `task1_mnist_pipeline.py` | MNIST 全流程腳本（含 stride/filter、L2 掃描、圖像分析） |
| `task2_cifar10_pipeline.py` | CIFAR-10 全流程腳本（含前處理、增強、消融與所有分析） |
| `easy_task*_*.py` | 為課程說明撰寫的簡化版腳本，不影響正式實驗結果 |
| `data/` | 下載的資料集與快取（會由腳本自動建立） |
| `artifacts/task*/` | 最佳模型、通道統計、訓練時間 (`training_durations.json`) |
| `logs/task*/` | CSV 訓練紀錄、TensorBoard 檔案 |
| `figures/task*/` | 學習曲線、混淆矩陣、正確/錯誤樣本、特徵圖、權重分佈等 |
| `reports/task*/` | 指標 CSV、前處理說明、feature map 筆記、summary JSON |
| `task1_report.md`, `task2_report.md` | 針對作業要求撰寫的詳細報告（繁中/英文混用） |

---

## 3. 模型與流程概述

### Task 1 – MNIST
- **模型**：3 個卷積區塊 + 全連接層，支援調整 stride/kernel 與 L2。  
- **訓練**：Adam (lr=1e-3)、batch size 128、ReduceLROnPlateau + EarlyStopping。  
- **輸出亮點**：`figures/task1/learning_curve_baseline.png`、`figures/task1/correct_vs_incorrect_baseline.png`、`reports/task1/l2_results.csv`。

### Task 2 – CIFAR-10
- **前處理**：標準化 + RandomCrop、Flip、Rotation、ColorJitter、Cutout 等增強，額外透過消融分析檢驗重要性。  
- **模型**：4 個卷積區塊（filters 64→512）搭配 AdamW 與 CosineRestart LR schedule。  
- **輸出亮點**：`figures/task2/confusion_matrix_baseline.png`、`figures/task2/correct_vs_incorrect_baseline.png`、`reports/task2/preprocessing.md`。

---

## 4. 執行方式

### 4.1 MNIST
```bash
cd HW2
python task1_mnist_pipeline.py --mode baseline           # 基準模型 + 視覺化
python task1_mnist_pipeline.py --mode stride_filter      # stride/kernel 掃描
python task1_mnist_pipeline.py --mode l2_study           # L2 正則化比較
python task1_mnist_pipeline.py --mode all                # 依序跑完全部（約 20 分鐘）
```

### 4.2 CIFAR-10
```bash
python task2_cifar10_pipeline.py --mode baseline              # 建議先跑一次
python task2_cifar10_pipeline.py --mode stride_filter         # Stride/Kernel 網格
python task2_cifar10_pipeline.py --mode l2_study              # L2 掃描
python task2_cifar10_pipeline.py --mode preprocessing_ablation  # 前處理消融（含 Cutout/Mixup）
python task2_cifar10_pipeline.py --mode all                   # 完整流程（>8 小時，慎用）
```

> **提示**：CIFAR-10 的 `--mode all` 會執行 22 次完整訓練。若僅需報告結果，可依需求挑選子模式降低耗時。

---

## 5. 產出檢視

- **報表**：`task1_report.md`、`task2_report.md` 內含文字說明、表格與圖檔連結。  
- **指標 CSV**：例如 `reports/task1/stride_filter_summary.csv`、`reports/task2/l2_results.csv`。  
- **圖像**：`figures/task*/learning_curve_*.png`、`confusion_matrix_*.png`、`correct_vs_incorrect_*.png`、`featuremaps_*.png`、`weights_*.png`。  
- **TensorBoard**：可透過 `tensorboard --logdir logs` 檢視訓練過程。

---

## 6. 重現流程建議

1. 依「環境準備」段落建立並驗證 TensorFlow GPU。  
2. 依序跑 MNIST 與 CIFAR-10 baseline 確認輸出目錄成立。  
3. 視需求執行 stride/filter、L2、前處理等子模式並整理 `reports/` 內的 CSV。  
4. 參考 `task1_report.md`、`task2_report.md` 寫作或補充個人觀察。  
5. 最後打包來源碼、報告與主要圖表，命名為 `hw2_<studentID>.zip` 符合作業繳交規範。

---

## 7. 版本控制與自訂設定

- `CONFIG` 區段可依需求修改；建議將自訂參數寫入 `configs/*.yaml` 並以 `--config` 指定，保留原始設定供比較。  
- 歡迎將新的實驗結果放入對應的 `reports/task*/` 與 `figures/task*/`，報告將引用該處的檔案。

---

如有任何實驗或環境設定疑問，歡迎直接檢視腳本註解或參考兩份報告中的補充說明。祝實驗順利！
   
