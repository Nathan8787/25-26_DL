"""
deep_learning_hw1_regression_pipeline
"""
from __future__ import annotations  # 允許型別註解中引用尚未定義的類別（Forward Reference）
import json  # 讀寫訓練摘要、標準化參數等 JSON 檔案
import math  # 進行批次大小切分、SVD 等運算時會用到的數學函式
import pickle  # 將模型權重序列化至硬碟，以便重複載入與評估
import random  # 控制 Python 標準庫層級的隨機性
from dataclasses import asdict, dataclass  # 方便定義資料結構並快速轉換為 dict
from pathlib import Path  # 管理 Windows/Linux/Mac 通用的路徑物件
from typing import Any, Dict, Iterable, List, Tuple  # 靜態型別註解用

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm # 進度條工具

# Matplotlib 預設會開啟互動視窗；在批次執行或遠端環境中改用 Agg 後端只輸出檔案即可。
plt.switch_backend("Agg")

CONFIG: Dict[str, Any] = {
    "seed": 29, #隨機種子碼
    #路徑管理
    "paths": {
        "dataset": Path("../datasets/2025_energy_efficiency_data.csv"), #資料集
        "processed_dir": Path("data/processed"),
        "artifacts_dir": Path("artifacts/regression"),
        "figures_dir": Path("figures"),
        "results_dir": Path("results"),
        "best_model": Path("artifacts/regression/best_model.pkl"),
        "training_history": Path("artifacts/regression/training_history.csv"),
        "feature_importance": Path("artifacts/regression/feature_importance.csv"),
        "feature_subset": Path("artifacts/regression/feature_subset_results.csv"),
        "summary": Path("results/regression_summary.json"),
    },
    #超參數設定
    "hyperparameters": {
        #學習步長
        "learning_rate": 0.01,
        # 總訓練輪數
        "epochs": 30000,
        #每個batch的數量
        "mini_batch_size": 32,
        "gradient_clip_value": 1.0,
        # 早停耐心值（validation 指標若連續該輪數未改善則停止）
        "early_stopping_patience": 5000,
    },
    # 連續型特徵欄位名稱（會做標準化）
    "continuous_features": [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "glazing_area",
    ],
    # orientation one-hot 編碼後的欄位名稱；順序與 categories 對齊
    "orientation_categories": [2, 3, 4, 5],
    "orientation_feature_names": [
        "orientation_is_2_North",
        "orientation_is_3_East",
        "orientation_is_4_South",
        "orientation_is_5_West",
    ],
     # gad one-hot 編碼後的欄位名稱；順序與 categories 對齊
    "gad_categories": [0, 1, 2, 3, 4, 5],
    "gad_feature_names": [
        "gad_is_0_Uniform",
        "gad_is_1_North",
        "gad_is_2_East",
        "gad_is_3_South",
        "gad_is_4_West",
        "gad_is_5_NorthEast",
    ],
}



@dataclass
class TrainingRecord:
    """紀錄單一 epoch 的訓練結果。"""

    epoch: int
    train_mse: float
    train_rms: float
    val_mse: float
    val_rms: float


def ensure_directories(paths: Iterable[Path]) -> None:
    """建立必要的輸出資料夾。"""

    for path in paths:
        if path.is_file():
            continue
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """設定 Python 與 NumPy 的隨機種子以確保結果可重現。"""

    random.seed(seed)
    np.random.seed(seed)


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """讀取原始能源效率資料並統一欄位名稱。"""

    rename_mapping = {
        "# Relative Compactness": "relative_compactness",
        "Surface Area": "surface_area",
        "Wall Area": "wall_area",
        "Roof Area": "roof_area",
        "Overall Height": "overall_height",
        "Orientation": "orientation",
        "Glazing Area": "glazing_area",
        "Glazing Area Distribution": "glazing_area_distribution",
        "Heating Load": "heating_load",
        "Cooling Load": "cooling_load", #雖然 cooling_load 不會用到，但還是改名了，後面會把它丟掉
    }
    df = pd.read_csv(path, dtype=float)
    df = df.rename(columns=rename_mapping)
    return df


def shuffle_and_split(df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    """將資料集依規格打散並切分為訓練、驗證、測試。

    流程：
    1) 依 seed 打散（sample(frac=1.0) 等價於隨機重排）。
    2) 先切出 75% 作為「train_full」，剩餘 25% 作為「test」。
    3) 在 train_full 中再切出 20% 作為「val」，其餘 80% 作為「train」。

    因此最終比例：
    - train ≈ 0.75 * 0.80 = 0.60（60%）
    - val   ≈ 0.75 * 0.20 = 0.15（15%）
    - test  ≈ 0.25（25%）

    註：
    - 使用 floor 保證整數索引切片；可能造成 ±1 樣本的分配差異，屬可接受範圍。
    """

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True) #打散
    n_samples = shuffled.shape[0]   #樣本數
    n_train_full = int(math.floor(n_samples * 0.75)) #75%當作train_full
    train_full = shuffled.iloc[:n_train_full].reset_index(drop=True) #前75%
    test = shuffled.iloc[n_train_full:].reset_index(drop=True) #後25%

    n_val = int(math.floor(train_full.shape[0] * 0.2)) #20%當作val
    val = train_full.iloc[:n_val].reset_index(drop=True) #前20%
    train = train_full.iloc[n_val:].reset_index(drop=True) #後80%

    return {"train": train, "val": val, "test": test} #回傳切分結果


def _one_hot_encode(
    values: pd.Series, categories: List[int], feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]: 
    
    """針對指定類別欄位進行 one-hot 編碼。

    參數：
    - values: 來源欄位（整數或可轉為整數的類別代碼）
    - categories: 允許的類別值列表（定義欄位順序）
    - feature_names: 對應於 categories 的輸出欄名（與順序一一對應）

    回傳：
    - encoded: 形狀 (N, len(categories)) 的 one-hot 矩陣，dtype=float
    - feature_names: 原樣回傳，便於外層串接欄名清單

    設計：
    - 先建立類別值到欄位索引的映射 index_map。
    - 對於每一筆資料，將對應類別的位置設為 1.0。
    - 若遇到不在 categories 的值，立刻拋錯，避免靜默錯誤。
    """
    index_map = {cat: idx for idx, cat in enumerate(categories)} #類別值到索引的映射
    encoded = np.zeros((values.shape[0], len(categories)), dtype=float) #初始化全零矩陣
    for row_idx, raw_value in enumerate(values.to_numpy()): 
        int_value = int(raw_value)
        if int_value not in index_map:
            raise ValueError(f"未知的類別值: {raw_value}")
        encoded[row_idx, index_map[int_value]] = 1.0
    return encoded, feature_names #回傳 one-hot 矩陣與欄名


def encode_and_standardize(
    splits: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]: 
    """執行 one-hot 編碼與連續特徵標準化，產出 NumPy 陣列。

    輸入：
    - splits: 由 shuffle_and_split 回傳的 dict，包含 "train"/"val"/"test" 三個 DataFrame

    輸出：
    - features: dict，鍵為 split 名稱，值為組合後的特徵矩陣（連續標準化 + 類別 one-hot）
    - targets:  dict，鍵為 split 名稱，值為目標變數（此處取 "heating_load"）
    - metadata: dict，包含標準化參數（mean、std）與欄名等資訊，可序列化保存

    設計重點：
    - 標準化（standardization）只在 train split 上 fit 均值/標準差，
      之後對 train/val/test 皆使用相同的 mean/std 進行 transform（避免資料洩漏）。
    - 連續特徵：CONFIG["continuous_features"]
    - 類別特徵：
        * orientation：使用 CONFIG["orientation_categories"] 與對應欄名
        * glazing_area_distribution：使用 CONFIG["gad_categories"] 與對應欄名
    - 合併順序：連續標準化後特徵 → orientation one-hot → gad one-hot
    - dtype：此處最終 features 以 float64 儲存（保持原實作）。
    """

    continuous_cols = CONFIG["continuous_features"]
    orient_categories = CONFIG["orientation_categories"]
    orient_feature_names = CONFIG["orientation_feature_names"]
    gad_categories = CONFIG["gad_categories"]
    gad_feature_names = CONFIG["gad_feature_names"]

    train_df = splits["train"].copy()

    train_continuous = train_df[continuous_cols].to_numpy(dtype=float)
    continuous_mean = train_continuous.mean(axis=0)
    continuous_std = train_continuous.std(axis=0, ddof=0)
    continuous_std[continuous_std == 0.0] = 1.0 #避免除以零

    features: Dict[str, np.ndarray] = {} #存放特徵矩陣
    targets: Dict[str, np.ndarray] = {} #存放目標變數

    feature_names: List[str] = [] #存放所有欄位名稱
    feature_names.extend(continuous_cols) 
    feature_names.extend(orient_feature_names)
    feature_names.extend(gad_feature_names)

    for split_name, df in splits.items(): #對 train/val/test 三個切分進行處理
        df_local = df.copy()
        continuous = df_local[continuous_cols].to_numpy(dtype=float)
        standardized = (continuous - continuous_mean) / continuous_std

        orientation_encoded, _ = _one_hot_encode( #針對 orientation 做 one-hot 編碼
            df_local["orientation"], orient_categories, orient_feature_names
        )
        gad_encoded, _ = _one_hot_encode( #針對 glazing_area_distribution 做 one-hot 編碼
            df_local["glazing_area_distribution"],
            gad_categories,
            gad_feature_names,
        )

        combined = np.hstack([standardized, orientation_encoded, gad_encoded]) #hstack來水平堆疊特徵
        features[split_name] = combined.astype(np.float64) #轉成 float64
        targets[split_name] = df_local[["heating_load"]].to_numpy(dtype=float) #目標變數

    metadata = { #存放標準化參數與欄名等資訊
        "continuous_mean": continuous_mean.tolist(),
        "continuous_std": continuous_std.tolist(),
        "feature_names": feature_names,
        "continuous_features": continuous_cols,
        "orientation_categories": orient_categories,
        "gad_categories": gad_categories,
        "orientation_feature_names": orient_feature_names,
        "gad_feature_names": gad_feature_names,
    }
    return features, targets, metadata



def build_mini_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int, seed: int
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """建立 mini-batch 迭代器以支援小批次 SGD。

    參數：
    - X: 特徵矩陣，形狀 (N, D)
    - y: 目標向量或矩陣，形狀 (N,) 或 (N, 1)
    - batch_size: 每批樣本數；最後一批可能小於 batch_size
    - seed: 控制每輪打散的隨機性（若每個 epoch 都呼叫一次，可傳遞不同 seed）

    行為：
    - 使用 NumPy Generator 的 permutation 產生一個隨機索引序列。
    - 依序切片產生 (X_batch, y_batch)。
    - 這是 generator；for 迴圈逐批消費，適合大資料避免一次載入全部到記憶體。
    """

    n_samples = X.shape[0] #樣本數
    rng = np.random.default_rng(seed) #建立隨機數產生器
    indices = rng.permutation(n_samples) #產生隨機索引序列
    for start in range(0, n_samples, batch_size): #每次取 batch_size 個樣本
        batch_idx = indices[start : start + batch_size] #切片索引
        yield X[batch_idx], y[batch_idx] #回傳該批次的 (X, y)


class NeuralNetwork:
    """Fully Connected Network。
    
    設計重點：
    - 僅用 NumPy，無自動微分：必須手寫 forward/backward。
    - 隱藏層使用 ReLU，輸出層為線性（適合回歸）。
    - 損失假設為 MSE：backward 的第一步直接用 dA = d(MSE)/d(y_pred)。
    - 參數初始化用 Glorot/Xavier uniform（對應 limit = sqrt(6/(fan_in+fan_out))）。
    """

    def __init__(self, layer_sizes: List[int], seed: int) -> None:
        """設定屬性與初始化權重。
        
        參數
        - layer_sizes: 各層神經元數量（含輸入與輸出），例如 [D, H1, H2, ..., C]
        - seed: 隨機種子（用於權重初始化）
        內部成員
        - self.weights: List[np.ndarray]，第 i 個元素形狀為 (layer_sizes[i], layer_sizes[i+1])
        - self.biases:  List[np.ndarray]，第 i 個元素形狀為 (1, layer_sizes[i+1])
        - self._cache:  在 forward(training=True) 時快取中間量 A、Z 供 backward 使用
        """
        self.layer_sizes = layer_sizes #各層神經元數量
        self.rng = np.random.default_rng(seed) #隨機數產生器
        self.weights: List[np.ndarray] = [] #權重矩陣列表
        self.biases: List[np.ndarray] = [] #Bias向量列表
        self._cache: Dict[str, List[np.ndarray]] | None = None #快取中間量
        self._initialize_parameters() #初始化權重與Bias

    def _initialize_parameters(self) -> None:
        """參數初始化（Glorot/Xavier uniform）
        流程：
        對於每個相鄰層 (fan_in, fan_out)：
        - W ~ U[-limit, limit]，limit = sqrt(6/(fan_in+fan_out))
        - b = 0
        """
        self.weights.clear() #清空權重列表
        self.biases.clear() #清空bias列表
        for idx in range(len(self.layer_sizes) - 1): #對每一層進行初始化
            fan_in = self.layer_sizes[idx]  #輸入神經元數
            fan_out = self.layer_sizes[idx + 1] #輸出神經元數
            limit = math.sqrt(6.0 / (fan_in + fan_out)) #Glorot/Xavier uniform limit
            # 權重矩陣形狀：(fan_in, fan_out)
            weight_matrix = self.rng.uniform(
                low=-limit, high=limit, size=(fan_in, fan_out)
            ) 
            # Bias向量形狀：(1, fan_out)；用行向量利於廣播
            bias_vector = np.zeros((1, fan_out), dtype=float) #初始化Bias向量為0
            self.weights.append(weight_matrix.astype(np.float64)) 
            self.biases.append(bias_vector.astype(np.float64))

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """前向傳播
        
        參數
        - X: 輸入，形狀 (N, D)；N 為batch size，D 為輸入維度
        - training: True 時會使用cache以供 backward 使用
        
        流程
        - 對每層做仿射變換：z = a @ W + b
        - 非最後一層套 ReLU：a = max(0, z)
        - 最後一層維持線性：a = z  （回歸輸出）
        
        回傳
        - a: 最終輸出，形狀 (N, C)
        - 若 training=True，快取：
          * "A": 每層的 activations（含輸入 X 作為 A[0]）
          * "Z": 每層的 pre-activations（不含輸入層）
        """
        
        activations = [X] #存放每層的激活值，初始為輸入X  A[0] = X
        zs: List[np.ndarray] = [] #存放每層的線性組合值 Z(不含輸入層)
        a = X #當前激活值，初始為輸入X
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)): #對每層進行前向傳播
            z = a @ W + b 
            zs.append(z)
            if layer_idx < len(self.weights) - 1:  
                a = np.maximum(0, z) #非最後一層使用ReLU
            else:
                a = z #最後一層維持線性
            activations.append(a)
        if training: #若是訓練模式，快取中間量
            self._cache = {"A": activations, "Z": zs}
        return a #回傳最終輸出

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """反向傳播（對應 MSE 損失）
        
        參數
        - y_true: 真值，形狀 (N, C) 或 (N, 1)
        - y_pred: 預測，形狀與 y_true 對齊
        
        梯度推導摘要
        - MSE = mean(||y_pred - y_true||^2)
          dL/dy_pred = 2*(y_pred - y_true)/N  → 記為 dA（最後一層的輸入梯度）
        - 對最末層（線性）：dZ = dA
        - 對隱藏層（ReLU）：dZ = dA * 1{Z>0}
        - 梯度計算：
          dW = A_prev^T @ dZ           （形狀與 W 相同）
          db = sum(dZ, axis=0, keepdims=True)   （形狀與 b 相同）
          dA_prev = dZ @ W^T           （回傳給前一層）
        
        回傳
        - grad_w: 與 self.weights 對應的梯度列表
        - grad_b: 與 self.biases  對應的梯度列表
        """
        if self._cache is None:
            raise RuntimeError("尚未執行 forward，無法進行 backward。")

        activations = self._cache["A"]  # A[0]=X, A[1]=第一層輸出, ..., A[L]=最終輸出
        zs = self._cache["Z"] # Z[0]=第一層 pre-activation, ..., Z[L-1]=最後一層 pre-activation

        grad_w = [np.zeros_like(W) for W in self.weights] #初始化梯度列表
        grad_b = [np.zeros_like(b) for b in self.biases]

        m = y_true.shape[0] #樣本數
        dA = (2.0 * (y_pred - y_true)) / m # dL/dA_L（A_L 即最終輸出），即對 MSE 的輸出梯度

        for layer_idx in reversed(range(len(self.weights))): # 從最後一層往前走
            if layer_idx == len(self.weights) - 1: 
                # 輸出層為線性：a_L = z_L → dZ = dA
                dZ = dA
            else:
                # 隱藏層為 ReLU：a = max(0, z) → dZ = dA * 1{z>0}
                relu_grad = (zs[layer_idx] > 0).astype(float)
                dZ = dA * relu_grad
            grad_w[layer_idx] = activations[layer_idx].T @ dZ  # dW = A_{l}^T @ dZ_{l+1}；A_{l} 為前一層輸出（activations[layer_idx]）
            grad_b[layer_idx] = np.sum(dZ, axis=0, keepdims=True) # db = 對樣本維度求和，保留形狀 (1, fan_out)
            dA = dZ @ self.weights[layer_idx].T  # 把梯度往前傳：dA = dZ @ W^T

        return grad_w, grad_b

    def apply_gradients(self, grad_w: List[np.ndarray], grad_b: List[np.ndarray], learning_rate: float) -> None:
        """以學習率做參數更新（等同於最簡 SGD step，不含動量/正規化）
        
        參數
        - grad_w, grad_b: 與 self.weights/biases 一一對應的梯度
        - learning_rate: 學習率 η
        
        更新規則
        - W := W - η * dW
        - b := b - η * db
        """
        for idx in range(len(self.weights)):
            self.weights[idx] -= learning_rate * grad_w[idx]
            self.biases[idx] -= learning_rate * grad_b[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """前向傳播網路進行預測"""
        return self.forward(X, training=False)

    def get_parameters(self) -> Dict[str, List[np.ndarray]]:
        """回傳目前模型參數（複本），便於保存或外部檢視或是早停時可以用
        
        回傳字典包含：
        - "weights": 權重列表（每個 shape 對應各層）
        - "biases":  Bias列表（每個 shape 為 (1, fan_out)）
        - "layer_sizes": 結構定義（含輸入與輸出維度）
        """
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "layer_sizes": self.layer_sizes,
        }

    def set_parameters(self, params: Dict[str, List[np.ndarray]]) -> None:
        """載入外部提供的參數        
        用途
        - 從檔案讀回最佳權重
        - 從別的實例同步參數
        """
        self.layer_sizes = params["layer_sizes"]
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_pred - y_true) ** 2)) #均方誤差
    rms = float(math.sqrt(mse)) #均方根誤差，主要看這個
    mae = float(np.mean(np.abs(y_pred - y_true))) #平均絕對誤差
    return {"mse": mse, "rms": rms, "mae": mae}


def gradient_check(model: NeuralNetwork, epsilon: float = 1e-5, sample_size: int = 10) -> float:
    """以有限差分對手刻梯度做數值檢查。

    目的
    - 驗證 backward 推導是否正確：比較「解析梯度」vs「數值梯度(答案)」。

    方法
    1) 造一個小批資料 X, y（固定 seed 可重現）。
    2) forward(training=True) → backward() 取「解析梯度」(grad_w_analytical, grad_b_analytical)。
    3) 隨機抽 sample_size 個權重元素 W[i,j]：
       - f_plus  = L(W[i,j]+ε)，f_minus = L(W[i,j]-ε)
       - numerical_grad ≈ (f_plus - f_minus) / (2ε)
       - relative_error = |ana - num| / max(|ana|, |num|)
       - 取最大 relative_error 作為本次檢查的結果
    4) 對每層Bias b 也做一個代表性的元素（b[0,0]）檢查（簡化版）。

    參數
    - model: 已定義結構與初始化完的 NeuralNetwork
    - epsilon: 差分步長 ε，太大誤差大，太小受浮點誤差影響
    - sample_size: 要抽幾個權重元素做檢查

    回傳
    - max_relative_error: 本次檢查的最大相對誤差。門檻設定1e-4 或更小視情況而定。
    """
    rng = np.random.default_rng(CONFIG["seed"])
    X = rng.normal(size=(4, model.layer_sizes[0])) # 小批 X，shape (4, D)
    y = rng.normal(size=(4, 1)) #對應4個樣本的目標值

    # 解析梯度：需先 forward(training=True) 以建立 cache，再 backward
    preds = model.forward(X, training=True) #前向傳播網路下得到的預測值
    grad_w_analytical, grad_b_analytical = model.backward(y, preds) #解析梯度
        
    # 收集所有權重元素的位置 (layer_idx, i, j)，隨機打亂並抽樣
    weight_positions = []                           # 將每個權重元素的索引三元組放進這個清單
    for layer_idx, W in enumerate(model.weights):   # 逐層走訪：layer_idx 是層索引，W 是該層權重矩陣
        rows = W.shape[0]                           # 權重矩陣的列數＝該層 fan_in
        cols = W.shape[1]                           # 權重矩陣的行數＝該層 fan_out
        for i in range(rows):                     
            for j in range(cols):                   
                weight_positions.append((layer_idx, i, j))  # 紀錄此權重元素的位置

    rng.shuffle(weight_positions)                   # 將所有位置隨機打亂，避免只檢查特定區塊
    weight_positions = weight_positions[:sample_size]  # 僅抽 sample_size 個位置做數值梯度檢查

    max_relative_error = 0.0                        # 紀錄全體檢查中的最大相對誤差

    # --- 權重的數值梯度檢查 ---
    for layer_idx, i, j in weight_positions:        # 逐個被抽樣的權重元素做檢查
        W = model.weights[layer_idx]                 # 取出該層的權重矩陣（別名參照）
        original = W[i, j]                           # 保存原始權重值，稍後要還原

        W[i, j] = original + epsilon                 # 擾動 +ε：θ+ = θ + ε
        plus = model.forward(X, training=False)      # 前向一次，training=False 避免覆寫 cache
        loss_plus = np.mean((plus - y) ** 2)         # 計算 f(θ+)

        W[i, j] = original - epsilon                 # 擾動 -ε：θ- = θ - ε
        minus = model.forward(X, training=False)     # 前向一次
        loss_minus = np.mean((minus - y) ** 2)       # 計算 f(θ-)

        W[i, j] = original                           # 還原該權重元素

        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)  # 中央差分近似數值梯度
        analytical_grad = grad_w_analytical[layer_idx][i, j]       # 對應的解析梯度（backward 算出的）

        if analytical_grad == 0 and numerical_grad == 0:           # 兩者皆為 0，跳過以免相對誤差分母為 0
            continue

        # 相對誤差：|g_ana - g_num| / max(|g_ana|, |g_num|)；避免除 0
        relative_error = abs(analytical_grad - numerical_grad) / max(
            abs(analytical_grad), abs(numerical_grad)
        )
        max_relative_error = max(max_relative_error, relative_error)  # 更新全域最大相對誤差

    # --- 偏置的數值梯度檢查（簡化：每層只檢 b[0,0] 一個元素）---
    for layer_idx, b in enumerate(model.biases):     # 逐層走訪Bias向量 b，形狀 (1, fan_out)
        original = b[0, 0]                           # 取第一個Bias元素作為代表

        b[0, 0] = original + epsilon                 # 擾動 +ε
        plus = model.forward(X, training=False)      # 前向
        loss_plus = np.mean((plus - y) ** 2)         # f(b+)

        b[0, 0] = original - epsilon                 # 擾動 -ε
        minus = model.forward(X, training=False)     # 前向
        loss_minus = np.mean((minus - y) ** 2)       # f(b-)

        b[0, 0] = original                           # 還原Bias

        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)   # 中央差分數值梯度
        analytical_grad = grad_b_analytical[layer_idx][0, 0]        # 解析梯度（backward）

        if analytical_grad == 0 and numerical_grad == 0:            # 兩者皆 0，跳過
            continue

        relative_error = abs(analytical_grad - numerical_grad) / max(
            abs(analytical_grad), abs(numerical_grad)
        )
        max_relative_error = max(max_relative_error, relative_error) # 更新最大相對誤差

    return max_relative_error                               # 回傳本次檢查中最糟的相對誤差（越小越好）

def preprocess_data(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """前處理總管：建目錄 → 讀檔 → 隨機切分 → 編碼 + 標準化。

    流程
    1) ensure_directories：確保輸出目錄存在（processed/artifacts/figures/results）。
    2) load_raw_dataset：讀 CSV 並正規化欄名。
    3) shuffle_and_split：依 seed 打散並切成 train/val/test（60/15/25）。
    4) 斷言三者筆數總和等於原始資料筆數（基本健檢）。
    5) encode_and_standardize：連續欄位用 train 的 mean/std 標準化；類別 one-hot。
       回傳三份：features（X）、targets（y）、metadata（縮放參數與欄名）。

    回傳
    - features: {"train": X_tr, "val": X_val, "test": X_test}
    - targets:  {"train": y_tr, "val": y_val, "test": y_test}
    - metadata: {"continuous_mean": ..., "feature_names": ...} 等資訊
    """
    paths = config["paths"]
    ensure_directories(
        [
            paths["processed_dir"],
            paths["artifacts_dir"],
            paths["figures_dir"],
            paths["results_dir"],
        ]
    )

    df = load_raw_dataset(paths["dataset"])
    splits = shuffle_and_split(df, seed=config["seed"])

    assert splits["train"].shape[0] + splits["val"].shape[0] + splits["test"].shape[0] == df.shape[0]

    features, targets, metadata = encode_and_standardize(splits) #執行one-hot encode與標準化
    return features, targets, metadata #回傳特徵、目標與metadata


def initialize_model(input_dim: int) -> NeuralNetwork:
    """根據輸入維度建立一個固定結構的前饋網路。

    結構
    - [input_dim, 8, 8, 4, 1]：三層隱藏層（ReLU），輸出層 1（回歸線性）
    - 權重初始化：NeuralNetwork 內部使用 Xavier uniform

    參數
    - input_dim: 輸入特徵數（X 的第二維）

    回傳
    - 已初始化的 NeuralNetwork 實例
    """
    layer_sizes = [input_dim, 16, 8, 4, 1]
    model = NeuralNetwork(layer_sizes=layer_sizes, seed=CONFIG["seed"])
    return model


def train_model(
    model: NeuralNetwork,
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    config: Dict[str, Any],
    perform_gradient_check: bool = True,
) -> Tuple[List[TrainingRecord], Dict[str, Any]]:
    """訓練主迴圈（前饋網路 + MSE），含梯度檢查、梯度裁剪、早停與最佳權重保存。

    參數
    - model: 前面寫好的NeuralNetwork 實例
    - features: {"train": X_tr, "val": X_val, "test": X_test}，此處用 train/val
    - targets:  {"train": y_tr, "val": y_val, "test": y_test}，此處用 train/val
    - config:   全域設定（超參數、路徑等）
    - perform_gradient_check: 是否在訓練前做一次有限差分梯度檢查

    回傳
    - history:  每個 epoch 的 TrainingRecord 列表（記錄 MSE/RMSE）
    - best_state: 最佳驗證 RMSE 時的參數快照（可用 set_parameters 還原）
    """

    # -------- 讀取超參數 --------
    hyper = config["hyperparameters"]
    lr = hyper["learning_rate"]
    epochs = hyper["epochs"]
    batch_size = hyper["mini_batch_size"]
    clip_value = hyper["gradient_clip_value"]
    patience = hyper["early_stopping_patience"]

    # -------- 取出訓練/驗證資料 --------
    train_X = features["train"]
    train_y = targets["train"]
    val_X = features["val"]
    val_y = targets["val"]

    # --------梯度檢查 --------
    if perform_gradient_check:
        rel_error = gradient_check(model)
        if rel_error > 1e-4:
            raise RuntimeError(f"梯度檢查失敗，最大相對誤差 {rel_error:.6f} 大於 1e-4。")

     # -------- 訓練狀態初始化 --------
    history: List[TrainingRecord] = []  # 紀錄每個 epoch 的指標
    best_state = model.get_parameters()  # 目前最佳權重（先存初始）
    best_val_rms = float("inf") # 目前最佳的驗證 RMSE
    epochs_since_best = 0 # 距離上次進步已過幾個 epoch

    # -------- 進度條（tqdm） --------
    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc="訓練進度",
        unit="epoch",
        leave=False,
    )

    # ================== 主要訓練迴圈 ==================
    for epoch in epoch_bar:
        # 每個 epoch 重建 batch 迭代器；seed 加上 epoch 讓每輪洗牌不同
        batch_iterator = build_mini_batches(train_X, train_y, batch_size, seed=CONFIG["seed"] + epoch)
        
        # -------- 小批次更新（SGD） --------
        for X_batch, y_batch in batch_iterator:
            preds = model.forward(X_batch, training=True)    # 前向，快取 A/Z
            grad_w, grad_b = model.backward(y_batch, preds)  # 解析梯度（MSE）

            # 梯度裁剪（數值穩定；避免單批爆炸）
            if clip_value is not None:
                grad_w = [np.clip(g, -clip_value, clip_value) for g in grad_w]
                grad_b = [np.clip(g, -clip_value, clip_value) for g in grad_b]

            model.apply_gradients(grad_w, grad_b, lr)      # 參數更新（最簡 SGD）

        # -------- epoch 結束：全量評估 train/val --------
        train_pred = model.predict(train_X)
        val_pred = model.predict(val_X)
        train_metrics = compute_metrics(train_y, train_pred)  # 回傳 dict：mse/rms/mae on training set
        val_metrics = compute_metrics(val_y, val_pred)     # 回傳 dict：mse/rms/mae on validation set

        # 紀錄本 epoch 結果
        record = TrainingRecord(
            epoch=epoch,
            train_mse=train_metrics["mse"],
            train_rms=train_metrics["rms"],
            val_mse=val_metrics["mse"],
            val_rms=val_metrics["rms"],
        )
        history.append(record)
        # 進度條即時顯示 RMSE
        epoch_bar.set_postfix(
            train_rms=f"{train_metrics['rms']:.4f}",
            val_rms=f"{val_metrics['rms']:.4f}",
        )

        # -------- 早停與最佳模型保存 --------
        if val_metrics["rms"] < best_val_rms - 1e-8:
            # 有實質進步（加上極小值避免浮點雜訊）
            best_val_rms = val_metrics["rms"]
            best_state = model.get_parameters() # 保存當前最佳參數
            epochs_since_best = 0 # 重設計數器
        else:
            epochs_since_best += 1 # 無進步，計數器 +1
            if epochs_since_best >= patience:
                break # 超過耐心值，停止訓練

    epoch_bar.close()

    model.set_parameters(best_state) # 訓練結束後還原到最佳參數
    return history, best_state #回傳訓練歷史與最佳參數快照


def save_training_history(history: List[TrainingRecord], path: Path) -> None:
    """把每個 epoch 的訓練紀錄（TrainingRecord）存成 CSV。

    參數
    - history: TrainingRecord 的列表（含 epoch、train/val 的 MSE/RMSE）
    - path: 輸出路徑，CONFIG["paths"]["training_history"]
    """
    # dataclass → dict（逐筆展開欄位），再組成 DataFrame
    df = pd.DataFrame([asdict(record) for record in history])
    # 不輸出索引欄
    df.to_csv(path, index=False)


def plot_learning_curve(history: List[TrainingRecord], path: Path) -> None:
    """繪製訓練/驗證 RMSE 隨 epoch 變化的學習曲線並存檔。

    參數
    - history: TrainingRecord 的列表
    - path: 圖檔儲存位置（.png 等）
    """
    # 取出 x 軸與兩條 y 軸序列
    epochs   = [record.epoch     for record in history]
    train_rms= [record.train_rms for record in history]
    val_rms  = [record.val_rms   for record in history]

    # 建立畫布
    plt.figure(figsize=(8, 5))
    # 訓練曲線：圓點標記
    plt.plot(epochs, train_rms, label="Train RMS", marker="o")
    # 驗證曲線：方塊標記
    plt.plot(epochs, val_rms,   label="Validation RMS", marker="s")
    # 標題與標籤
    plt.xlabel("Epoch")
    plt.ylabel("RMS")
    plt.title("Learning Curve")
    plt.legend()
    # 網格線
    plt.grid(True, linestyle="--", alpha=0.5)
    # 自動調整邊距避免標籤被裁切
    plt.tight_layout()
    # 只存檔不顯示（後端已設為 Agg）
    plt.savefig(path)
    plt.close()

def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """繪製 y_true 與 y_pred 的散點圖，並畫 y=x 參考線。

    參數
    - y_true: 真值，shape (N, 1) 或 (N,)
    - y_pred: 預測，shape 與 y_true 對齊
    - title: 圖標題
    - path: 輸出圖檔路徑
    """
    plt.figure(figsize=(6, 6))
    # 散點：每一筆 (y_true, y_pred)
    plt.scatter(y_true, y_pred, alpha=0.6)

    # 計算對角參考線範圍（覆蓋真值與預測的最小/最大範圍）
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    # 畫 y = x 的紅色虛線，做為理想預測的參考
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="y = x")

    # 標籤與網格
    plt.xlabel("True Heating Load")
    plt.ylabel("Predicted Heating Load")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate_model(
    model: NeuralNetwork,
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """以目前模型在 train/val/test 上做推論，回傳 MSE/RMSE/MAE，並輸出圖表。

    輸入
    - model: 已載入最佳權重或當前狀態的模型
    - features/targets: 三個 split 的 X/y

    產出
    - results: {"train": {"mse":..,"rms":..,"mae":..}, "val": {...}, "test": {...}}
    - 同時在 figures_dir 輸出：
        1) 每個 split 的 pred vs actual 散點圖
        2) 訓練集與測試集的「預測與真值折線圖」（方便逐樣本觀察）
    """
    results: Dict[str, Dict[str, float]] = {}
    for split_name in ["train", "val", "test"]:
        preds = model.predict(features[split_name])
        metrics = compute_metrics(targets[split_name], preds)
        results[split_name] = metrics

        # 原本的散點圖
        fig_path = CONFIG["paths"]["figures_dir"] / f"regression_pred_vs_actual_{split_name}.png"
        plot_prediction_scatter(
            targets[split_name],
            preds,
            title=f"{split_name.capitalize()} Predictions vs Actual",
            path=fig_path,
        )

    # 新增：訓練集與測試集的預測 vs 真實值折線圖
    def plot_pred_vs_label(y_true, y_pred, title, path):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(y_true, label='label', color='blue')
        plt.plot(y_pred, label='predict', color='red')
        plt.xlabel('#th case')
        plt.ylabel('heating load')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # 用 best model 畫
    train_pred = model.predict(features['train']).flatten()
    train_true = targets['train'].flatten()
    plot_pred_vs_label(train_true, train_pred,
                      'prediction for training data',
                      CONFIG["paths"]["figures_dir"] / "regression_pred_vs_label_train.png")

    test_pred = model.predict(features['test']).flatten()
    test_true = targets['test'].flatten()
    plot_pred_vs_label(test_true, test_pred,
                      'prediction for test data',
                      CONFIG["paths"]["figures_dir"] / "regression_pred_vs_label_test.png")

    return results


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_best_model(path: Path) -> NeuralNetwork:
    """從 pickle 檔載入以 get_parameters() 存下的最佳權重，回復成模型。

    流程
    1) 讀檔得到 params（應為 dict，含 weights/biases/layer_sizes）
    2) 以同樣的 layer_sizes 建立模型實例
    3) set_parameters(params) 將權重/偏置灌回模型

    回傳
    - 可直接用於 predict/evaluate 的 NeuralNetwork 實例
    """
    with path.open("rb") as fp:
        params = pickle.load(fp)
    model = NeuralNetwork(layer_sizes=params["layer_sizes"], seed=CONFIG["seed"])
    model.set_parameters(params)
    return model


def permutation_importance(
    model: NeuralNetwork,
    base_features: np.ndarray,
    base_targets: np.ndarray,
    feature_names: List[str],
    group_mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """以「置換重要度（Permutation Importance）」評估特徵群對效能的影響。

    觀念：
    - 基準：先計算未擾動時的驗證誤差（baseline）。
    - 對某一群特徵的每一欄，將其值「隨機重排」(permutation，不改變邊際分佈)，再重新預測。
    - 若該群特徵重要，亂序會使誤差上升；誤差上升量即定義為該群的重要度。

    參數
    - model: 訓練完成可做 predict 的模型
    - base_features: 驗證用 X（shape: (N, D)）
    - base_targets: 驗證用 y（shape: (N, 1) 或 (N,)）
    - feature_names: D 維特徵對應的欄名列表（與 base_features 欄位順序一致）
    - group_mapping: {"G1": [name1, name2, ...], ...} 特徵群對應的欄名清單

    回傳
    - DataFrame 欄位：
      ["group_id", "features", "baseline_rms", "permuted_rms", "importance"]
      其中 importance = permuted_rms - baseline_rms（越大越重要）
    """
    
    # Baseline performance
    baseline_preds = model.predict(base_features)
    baseline_metrics = compute_metrics(base_targets, baseline_preds)
    baseline_rms = baseline_metrics["rms"]


    rng = np.random.default_rng(CONFIG["seed"])
    results = []
    # 2) 對每個特徵群做一次「整群置換」的重要度評估
    for group_id, features_in_group in group_mapping.items():
        # 找到該群特徵在 base_features 中的欄位索引
        columns = [feature_names.index(name) for name in features_in_group]
        permuted = base_features.copy()  # 建一份可修改的副本

        # 3) 對群內每一欄進行「就地隨機重排」
        for col in columns:
             # permutation 是逐欄位洗牌，保持該欄邊際分佈但破壞與其他欄和目標的關聯
            permuted[:, col] = rng.permutation(permuted[:, col])
            
        # 4) 以置換後的資料推論，計算誤差
        permuted_preds = model.predict(permuted)
        permuted_rms = compute_metrics(base_targets, permuted_preds)["rms"]
        # 5) 重要度＝誤差上升量
        importance = permuted_rms - baseline_rms
        results.append(
            {
                "group_id": group_id,
                "features": features_in_group,
                "baseline_rms": baseline_rms,
                "permuted_rms": permuted_rms,
                "importance": importance,
            }
        )
    # 6) 彙整並依重要度排序（降序：越重要排越前）
    df = pd.DataFrame(results)
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df


def plot_feature_importance(df: pd.DataFrame, path: Path) -> None:
    """畫出置換重要度的橫條圖（大者重要）。

    參數
    - df: permutation_importance 的結果，需含 "group_id" 與 "importance"
    - path: 圖檔輸出路徑
    """
    
    # 橫條圖：x 軸為重要度（RMS 增量），y 軸為群組 ID
    plt.figure(figsize=(8, 6))
    plt.barh(df["group_id"], df["importance"])
    plt.xlabel("RMS increase after permutation")
    plt.ylabel("Feature group")
    plt.title("Permutation Importance (Validation set)")
    plt.gca().invert_yaxis()   # 讓最重要（數值最大）顯示在最上面
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def select_features(
    feature_dict: Dict[str, np.ndarray],
    feature_names: List[str],
    selected_names: List[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """依照選定的特徵名稱子集，從各 split 的 X 抽取對應欄位。

    參數
    - feature_dict: {"train": X_tr, "val": X_val, "test": X_test}
    - feature_names: 全部特徵欄名（對應原 X 的欄位順序）
    - selected_names: 欲保留的特徵名清單（可重複或無序）

    回傳
    - subset: 與 feature_dict 結構相同，但只含選定欄的子矩陣
    - ordered_names: 按原 feature_names 順序排序且去重後的特徵名
    """
    # 1) 依原始欄位順序排序並去重（確保子集欄位順序與原資料一致）
    ordered_names = sorted(set(selected_names), key=lambda name: feature_names.index(name))
    # 2) 轉成實際欄位索引
    indices = [feature_names.index(name) for name in ordered_names]

    subset: Dict[str, np.ndarray] = {}
    # 3) 對 train/val/test 都切同樣的欄位
    for split_name, data in feature_dict.items():
        subset[split_name] = data[:, indices]
    return subset, ordered_names


def evaluate_feature_subsets(
    base_features: Dict[str, np.ndarray],
    base_targets: Dict[str, np.ndarray],
    feature_names: List[str],
    group_importance: pd.DataFrame,
) -> pd.DataFrame:
    """依特徵群的重要度生成多個子集合，逐一重訓與評估，彙整比較表。

    流程概述
    1) 定義群組 → 實際欄名的對映（與 permutation_importance 使用一致）。
    2) 依重要度排序，建立多個子集合方案（Top-k、All、All-minus-某群）。
    3) 對每個子集合：
       - 擷取對應特徵欄位
       - 以相同超參數訓練模型（可關閉梯度檢查以節省時間）
       - 保存該子集合的訓練歷史
       - 以 train/val/test 評估，收集 RMSE
    4) 回傳各子集合的結果表（subset_id、included_groups、num_features、三組 RMSE）

    參數
    - base_features/base_targets: 原始三個 split 的 X/y
    - feature_names: 全部特徵欄名
    - group_importance: permutation_importance 的結果，用以確定群組排序
    """
    # 1) 群組定義（與 encode_and_standardize 的輸出欄名一致）
    group_map = {
        "G1": ["relative_compactness"],
        "G2": ["surface_area"],
        "G3": ["wall_area"],
        "G4": ["roof_area"],
        "G5": ["overall_height"],
        "G6": ["glazing_area"],
        "G7": CONFIG["orientation_feature_names"],  # 多個 one-hot 欄
        "G8": CONFIG["gad_feature_names"],          # 多個 one-hot 欄
    }

    # 2) 依重要度降序排列群組（permutation_importance 已經排序）
    sorted_groups = list(group_importance["group_id"])

    # 3) 預設要跑的子集合：Top-1/3/5 與 All
    subsets_to_run = [
        ("Top-1", sorted_groups[:1]),
        ("Top-3", sorted_groups[:3]),
        ("Top-5", sorted_groups[:5]),
        ("All",   sorted_groups),
    ]

    # 4) 額外方案：若 one-hot 大群（G7/G8）在末位，另外跑「All-minus-該群」觀察其貢獻，
    # 檢查把大型 one-hot 群（G7=orientation、G8=gad）整組拿掉，模型表現是否更好或幾乎不變。
    # 用來驗證「PI 排名靠後」是否代表可刪、或是 PI 被稀釋（共線、維度高）而低估了它們。
    low_rank_groups = sorted_groups[-2:]  # 最後兩名
    for group_id in ["G7", "G8"]:
        if group_id in low_rank_groups:
            remaining = [g for g in sorted_groups if g != group_id]
            subsets_to_run.append((f"All-minus-{group_id}", remaining))

    results = []

    # 5) 逐個子集合重訓與評估（使用 tqdm 顯示進度）
    for subset_id, groups in tqdm(subsets_to_run, desc="特徵子集合重訓", unit="subset", leave=False):
        # 5.1 展開這個子集合所包含的所有實際欄名（將群組映射到欄名）
        selected_features: List[str] = []
        for g in groups:
            selected_features.extend(group_map[g])

        # 5.2 從 base_features 擷取子集欄位，並取得排序後欄名
        subset_data, ordered_names = select_features(base_features, feature_names, selected_features)
        subset_targets = base_targets  # y 不變（回歸目標）

        # 5.3 以子集的輸入維度建立新模型並訓練
        model = initialize_model(input_dim=len(ordered_names))
        history, best_state = train_model(
            model,
            subset_data,
            subset_targets,
            CONFIG,
            perform_gradient_check=False,  # 關閉以加速多次重訓流程
        )

        # 5.4 保存訓練歷史（每個子集合一份）
        train_hist_path = CONFIG["paths"]["artifacts_dir"] / f"training_history_{subset_id}.csv"
        save_training_history(history, train_hist_path)

        # 5.5 還原最佳權重並做最終評估
        model.set_parameters(best_state)
        metrics = evaluate_model(model, subset_data, subset_targets)

        # 5.6 紀錄此子集合的關鍵資訊與三組 RMSE
        results.append(
            {
                "subset_id": subset_id,           # 子集合名稱（Top-3/All-minus-G7 等）
                "included_groups": groups,        # 這個子集合包含哪些群組
                "num_features": len(ordered_names),  # 實際特徵數量
                "train_rms": metrics["train"]["rms"],
                "val_rms":   metrics["val"]["rms"],
                "test_rms":  metrics["test"]["rms"],
            }
        )

    # 6) 回傳彙整表（可另存 CSV 以便對照不同子集合的表現）
    df = pd.DataFrame(results)
    return df

def run_feature_selection(
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    best_model_path: Path,
) -> None:
    """整體特徵選擇程序入口：先做群組置換重要度，再依排名生成多個子集合重訓評估。

    參數
    - features: {"train": X_tr, "val": X_val, "test": X_test}
    - targets:  {"train": y_tr, "val": y_val, "test": y_test}
    - metadata: 需至少包含 "feature_names"（X 欄位順序對應的名稱）
    - best_model_path: 先前以 get_parameters() 存的最佳模型權重（pickle）

    輸出
    - CSV：特徵群置換重要度（CONFIG["paths"]["feature_importance"]）
    - 圖檔：置換重要度橫條圖（figures/regression_feature_importance.png）
    - CSV：不同特徵子集合的表現（CONFIG["paths"]["feature_subset"]）
    - 圖檔：子集合在驗證集的 RMSE 長條圖（figures/regression_subset_performance.png）
    """

    # ---------------- 1) 定義「群組 → 實際欄名」的映射 ----------------
    # 說明：
    # - 單一連續特徵各自一組（G1~G6）
    # - 多欄 one-hot 的類別群（orientation, gad）各自成一組（G7、G8）
    #   目的：用「群組置換」減少共線與 one-hot 分裂造成的重要度低估。
    group_mapping = {
        "G1": ["relative_compactness"],
        "G2": ["surface_area"],
        "G3": ["wall_area"],
        "G4": ["roof_area"],
        "G5": ["overall_height"],
        "G6": ["glazing_area"],
        "G7": CONFIG["orientation_feature_names"],
        "G8": CONFIG["gad_feature_names"],
    }

    # ---------------- 2) 載入已訓練的最佳模型（固定模型做解釋） ----------------
    model = load_best_model(best_model_path)

    # 僅用驗證集做特徵重要度與子集合選擇，避免測試集洩漏
    val_features = features["val"]
    val_targets  = targets["val"]

    # ---------------- 3) 群組置換重要度（Permutation Importance, on validation） ----------------
    # 觀念：打亂某群特徵的列順序（破壞與 y 的對應），前向重算 RMSE，
    #       「RMSE 增量 = 該群重要度」。不重訓模型，只做推論。
    importance_df = permutation_importance(
        model,
        val_features,
        val_targets,
        metadata["feature_names"],  # 欄名順序需對齊 X 的欄位
        group_mapping,
    )

    # 保存重要度表格（便於報告與追蹤）
    importance_path = CONFIG["paths"]["feature_importance"]
    importance_df.to_csv(importance_path, index=False)

    # 視覺化：群組重要度橫條圖（大者重要）
    plot_feature_importance(
        importance_df,
        CONFIG["paths"]["figures_dir"] / "regression_feature_importance.png",
    )

    # ---------------- 4) 依重要度排序生成多個特徵子集合並重訓評估 ----------------
    # evaluate_feature_subsets 內部會：
    # - 依群組排名建立 Top-k / All / 以及 All-minus-大型群（若其排名靠後）
    # - 為每個子集合擷取欄位 → 重新初始化模型 → 訓練（可關閉梯檢）→ 在 train/val/test 評估
    # - 回傳每個子集合的 RMSE 彙整表
    subset_df = evaluate_feature_subsets(
        features,
        targets,
        metadata["feature_names"],
        importance_df,
    )
    # 保存子集合表現（subset_id、included_groups、num_features、三組 RMSE）
    subset_df.to_csv(CONFIG["paths"]["feature_subset"], index=False)

    # ---------------- 5) 視覺化：不同子集合在驗證集的 RMSE ----------------
    plt.figure(figsize=(8, 5))
    plt.bar(subset_df["subset_id"], subset_df["val_rms"])  # 以驗證集 RMSE 作比較基準
    plt.xlabel("Subset ID")
    plt.ylabel("Validation RMS")
    plt.title("Feature Subset Performance")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(CONFIG["paths"]["figures_dir"] / "regression_subset_performance.png")
    plt.close()

def run_pipeline() -> None:
    """固定執行完整流程：preprocess → train → evaluate → feature_selection。"""
    set_global_seed(CONFIG["seed"])               # 固定隨機性
    paths = CONFIG["paths"]                       # 取出路徑設定

    # 1) 準備輸出目錄
    ensure_directories([
        paths["processed_dir"],
        paths["artifacts_dir"],
        paths["figures_dir"],
        paths["results_dir"],
    ])

    # 進度條固定四階段
    stage_bar = tqdm(total=4, desc="流程總進度", unit="stage", leave=False)

    # 2) 前處理：讀檔→切分→one-hot→標準化；回傳 X/y 與中介資訊
    tqdm.write("[INFO] 開始資料前處理…")
    features, targets, metadata = preprocess_data(CONFIG)
    stage_bar.update(1)

    # 3) 訓練：含梯度檢查、梯度裁剪、早停；保存最佳權重與學習曲線
    tqdm.write("[INFO] 開始模型訓練…")
    model = initialize_model(input_dim=features["train"].shape[1])
    history, best_state = train_model(model, features, targets, CONFIG, perform_gradient_check=True)
    save_training_history(history, paths["training_history"])
    plot_learning_curve(history, paths["figures_dir"] / "regression_learning_curve.png")
    with paths["best_model"].open("wb") as fp:
        pickle.dump(best_state, fp)               # 保存 get_parameters() 回來的快照
    stage_bar.update(1)

    # 4) 評估：載入最佳權重，在 train/val/test 上計算指標與輸出圖
    tqdm.write("[INFO] 開始模型評估…")
    best_model = load_best_model(paths["best_model"])
    results = evaluate_model(best_model, features, targets)
    save_json(results, paths["summary"])
    stage_bar.update(1)

    # 5) 特徵選擇：群組 PI → 依排名重訓多個子集合 → 輸出表與圖
    tqdm.write("[INFO] 開始特徵選取與重訓分析…")
    run_feature_selection(features, targets, metadata, paths["best_model"])
    stage_bar.update(1)

    stage_bar.close()


if __name__ == "__main__":
    run_pipeline()   