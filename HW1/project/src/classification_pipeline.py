"""
classification_pipeline
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
from tqdm.auto import tqdm 

# Matplotlib 預設會開啟互動視窗；在批次執行或遠端環境中改用 Agg 後端只輸出檔案即可。
plt.switch_backend("Agg")


CONFIG: Dict[str, Any] = {
    # 固定亂數種子，確保訓練結果可重現（含 Python random、NumPy、模型初始化）。
    "seed": 666666,
    # 統一管理所有檔案路徑，後續改路徑只需在此調整。
    "paths": {
        "dataset": Path("../datasets/2025_ionosphere_data.csv"),
        "processed_dir": Path("data/processed/classification"),
        "artifacts_dir": Path("artifacts/classification"),
        "figures_dir": Path("figures"),
        "results_dir": Path("results"),
        "best_model": Path("artifacts/classification/best_model.pkl"),
        "training_history": Path("artifacts/classification/training_history.csv"),
        "latent_dir": Path("artifacts/classification/latents"),
        "latent_comparison": Path("artifacts/classification/latent_comparison.csv"),
        "summary": Path("results/classification_summary.json"),
    },
    # 神經網路訓練相關的超參數設定。
    "hyperparameters": {
        "learning_rate": 0.01,
        "epochs": 3000,
        "mini_batch_size": 32,
        "gradient_clip_value": 1.0,
        "early_stopping_patience": 300,
        "early_stopping_min_delta": 1e-6 # 早停的最小改善幅度
    },
    # 進行 latent 分析時，會針對這些隱藏層維度重新訓練模型，比較特徵表示力。
    "latent_layer_sizes": [4,16, 32, 64, 128],
}


@dataclass
class TrainingRecord:
    """紀錄單一 epoch 的訓練結果。"""
    epoch: int
    train_loss: float
    train_accuracy: float
    train_error_rate: float
    val_loss: float
    val_accuracy: float
    val_error_rate: float


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


def load_dataset(path: Path) -> pd.DataFrame:
    """讀取 ionosphere 資料集並補上欄位名稱。"""
    df = pd.read_csv(path, header=None)
    feature_names = [f"f{i}" for i in range(df.shape[1] - 1)]
    df.columns = feature_names + ["label"]
    return df

def split_dataset(df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    """按照 64%/16%/20% 的比例切分訓練、驗證與測試資料。

    流程：
    1. 使用固定種子 shuffle（`sample(frac=1.0)`）。
    2. 前 80% 作為暫時的訓練+驗證集合，後 20% 固定為測試集。
    3. 在訓練+驗證集合再切 20% 當驗證集，得到最終 64/16/20 的切分比例。
    """
    # 以 seed 重現隨機打散；重置 index 方便後續切片
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 總樣本數
    n_total = shuffled.shape[0]

    # 先切出 80%（train+val）與 20%（test）
    n_train_val = int(math.floor(n_total * 0.8))   # floor 確保整數索引安全
    train_val = shuffled.iloc[:n_train_val].reset_index(drop=True)
    test = shuffled.iloc[n_train_val:].reset_index(drop=True)

    # 在 train_val 當中，再拿 20% 當驗證：0.8 * 0.2 = 0.16
    n_val = int(math.floor(train_val.shape[0] * 0.2))
    val = train_val.iloc[:n_val].reset_index(drop=True)
    train = train_val.iloc[n_val:].reset_index(drop=True)

    # 以 dict 形式回傳三個切分
    return {"train": train, "val": val, "test": test}


def standardize_features(
    splits: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """進行 z-score 標準化並轉為 NumPy 陣列。

    輸入
    ----
    splits : dict
        `split_dataset` 的輸出，含 train/val/test 三個 DataFrame。

    回傳
    ----
    features : Dict[str, np.ndarray]
        三個資料切分的標準化特徵矩陣。
    targets : Dict[str, np.ndarray]
        二元標籤（"g"→1, "b"→0）。
    metadata : Dict[str, Any]
        儲存 mean/std 與 label 對映，供後續重建 scaler。

    """
    # 除 "label" 外其餘皆視為特徵欄
    feature_cols = [col for col in splits["train"].columns if col != "label"]

    # 以訓練集估計標準化參數（只用 train，避免洩漏）
    train_features = splits["train"][feature_cols].to_numpy(dtype=float)
    mean = train_features.mean(axis=0)                 # 每一欄的均值
    std = train_features.std(axis=0, ddof=0)          # 母體標準差（ddof=0）
    std[std == 0.0] = 1.0                              # 常數欄避免除以 0

    features: Dict[str, np.ndarray] = {}
    targets: Dict[str, np.ndarray] = {}

    # 文字標籤轉數值；可視需要擴充
    label_map = {"g": 1.0, "b": 0.0}

    # 對三個 split 套用同一組 mean/std，並轉成 float64
    for split_name, df in splits.items():
        feat = df[feature_cols].to_numpy(dtype=float)
        standardized = (feat - mean) / std            # z-score
        features[split_name] = standardized.astype(np.float64)

        # 轉換 label → 浮點數，再 reshape 成 (N,1)
        labels = df["label"].map(label_map).to_numpy(dtype=float)
        targets[split_name] = labels.reshape(-1, 1)

    # 保存 scaler 與欄名等中介資訊，便於重現與部署
    metadata = {
        "feature_names": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "label_map": label_map,
    }
    return features, targets, metadata

def build_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """生成 mini-batch，供 SGD/mini-batch GD 使用。

    參數
    - X: 特徵矩陣，shape = (N, D)
    - y: 標籤向量/矩陣，shape = (N, 1) 或 (N,)
    - batch_size: 每批樣本數；最後一批可能小於此值
    - seed: 隨機種子；控制洗牌順序以保重現

    產出 (yield)
    - (X_batch, y_batch)：連續吐出批次資料，不一次載入到記憶體

    設計重點
    - 以 np.random.Generator(permutation) 產生 0..N-1 的亂序索引。
    - 以步長 batch_size 切片索引；切到尾端自然會得到較小批次。
    - 使用 generator（yield）避免複製整份資料，節省記憶體。
    - 想每個 epoch 都洗不同順序：外部每輪改 seed（如 seed+epoch）。
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]



class NeuralNetworkClassifier:
    """以純 NumPy 實現的二元分類前饋式神經網路。

    設計重點
    --------
    - 隱藏層使用 ReLU（非線性、計算簡單），輸出層使用 sigmoid（輸出機率）。
    - 權重初始化採 Glorot/Xavier uniform，降低梯度爆炸或消失風險。
    - 透過 `_cache` 儲存 forward 的中間結果（A/Z），供 backward 使用。
    - 內建 `get_latent_output`，可在特定層擷取 latent representation 做視覺化或後續分析。

    介面與假設
    ----------
    - 輸入 X 形狀 (N, D)。
    - 標籤 y 形狀 (N, 1)，取值 0 或 1（float）。
    - 損失函數假設為 Binary Cross-Entropy；對應的輸出層導數在實作中使用
      dA = (y_pred - y_true) / N（相容於 sigmoid + BCE 的常見組合）。
    - layer_sizes: List[int]，例如 [D, 128, 64, 16, 1]。
    """

    def __init__(self, layer_sizes: List[int], seed: int) -> None:
        # 保存網路結構，如 [D, 128, 64, latent, 1]
        self.layer_sizes = layer_sizes
        # 可重現的亂數產生器（初始化權重/打亂等都可用）
        self.rng = np.random.default_rng(seed)
        # 權重與Bias容器；每層對應一個 ndarray
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        # forward(training=True) 會快取每層的 A/Z，供 backward 使用
        self._cache: Dict[str, List[np.ndarray]] | None = None
        # 可選：保存某一層（倒數第二層）的 latent 表徵，方便取用
        self._latent_output: np.ndarray | None = None
        # 初始化所有層的參數（W, b）
        self._init_parameters()

    def _init_parameters(self) -> None:
        """依 Glorot (Xavier) uniform 初始化各層權重與偏置。

        對於每個相鄰層 (fan_in, fan_out)：
        - limit = sqrt(6 / (fan_in + fan_out))
        - W ~ Uniform[-limit, +limit]，shape = (fan_in, fan_out)
        - b = 0，shape = (1, fan_out)
        """
        self.weights.clear()
        self.biases.clear()
        for idx in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[idx]
            fan_out = self.layer_sizes[idx + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            # 權重用一個合理範圍的均勻分佈初始化（浮點 64 提升數值穩定）
            weight = self.rng.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
            bias = np.zeros((1, fan_out), dtype=float)
            self.weights.append(weight.astype(np.float64))
            self.biases.append(bias.astype(np.float64))

    # ---- 基本激活與梯度 ----
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        # ReLU(x) = max(0, x)
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        # ReLU'(x) = 1{x > 0}，在 0 點不可導，取 0 或 1 均可；慣例用 0。
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Sigmoid(x) = 1 / (1 + e^{-x})；二元分類輸出機率
        # 若 x 絕對值極大會有溢位風險；此處保持簡潔，必要時可做 clip。
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """前向傳播。

        參數
        ----
        X : np.ndarray
            形狀 (N, D) 的特徵矩陣。
        training : bool
            設為 True 會將 activations (A) 與 pre-activations (Z) 存入 cache。

        回傳
        ----
        np.ndarray
            最終輸出（sigmoid 後的機率，形狀 (N, 1)）。
        """
        activations = [X]            # A^0 = X，方便對齊索引
        zs: List[np.ndarray] = []    # 保存每層 Z = A @ W + b
        latent_output: np.ndarray | None = None  # 供外部存取的中間層輸出

        a = X
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            # 仿射變換 Z^l = A^{l-1} W^l + b^l
            z = a @ W + b
            zs.append(z)

            # 最後一層使用 sigmoid，其他層使用 ReLU
            if idx == len(self.weights) - 1:
                a = self._sigmoid(z)
            else:
                a = self._relu(z)
                # 這裡選擇把「倒數第二層」的激活作為 latent 表徵
                if idx == len(self.weights) - 2:
                    latent_output = a

            activations.append(a)

        # 訓練模式下保存 cache，供 backward 鏈式法則使用
        if training:
            self._cache = {"A": activations, "Z": zs}
            self._latent_output = latent_output
        else:
            # 推論時也保存最新一次的 latent，方便外部取用
            self._latent_output = latent_output

        return a  # 機率 (N, 1)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """反向傳播，計算每一層的梯度。

        假設輸出層採用 sigmoid 並以 Binary Cross-Entropy 損失：
        dA_L = (y_pred - y_true) / N。
        之後逐層傳回：
        - dZ_l = dA_l * g'(Z_l)（最後一層線性到 sigmoid 的簡化已吸收於 dA_L）
        - dW_l = A_{l-1}^T @ dZ_l
        - db_l = sum(dZ_l)
        - dA_{l-1} = dZ_l @ W_l^T
        """
        if self._cache is None:
            raise RuntimeError("尚未呼叫 forward，無法進行 backward。")

        activations = self._cache["A"]  # 向前的輸入/輸出序列
        zs = self._cache["Z"]           # 每層 Z

        # 與 weights/biases 同形狀的梯度容器
        grad_w = [np.zeros_like(W) for W in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        m = y_true.shape[0]              # 批大小 N
        # Sigmoid + BCE 常見簡化：dA_L = (y_pred - y_true) / N
        dA = (y_pred - y_true) / m

        # 從最後一層往前（反向）
        for idx in reversed(range(len(self.weights))):
            if idx == len(self.weights) - 1:
                # 最後一層：dZ 等於 dA（已吸收 sigmoid + BCE 的導數簡化）
                dZ = dA
            else:
                # 隱藏層：乘上 ReLU 導數遮罩
                dZ = dA * self._relu_grad(zs[idx])

            # 參數梯度：dW = A^{l-1}^T @ dZ；db 為對列方向求和
            grad_w[idx] = activations[idx].T @ dZ
            grad_b[idx] = np.sum(dZ, axis=0, keepdims=True)

            # 傳回一層：dA^{l-1} = dZ @ W^T
            dA = dZ @ self.weights[idx].T

        return grad_w, grad_b

    def apply_gradients(self, grad_w: List[np.ndarray], grad_b: List[np.ndarray], lr: float) -> None:
        """以最簡 SGD 規則更新參數：W -= lr * dW；b 同理。"""
        for idx in range(len(self.weights)):
            self.weights[idx] -= lr * grad_w[idx]
            self.biases[idx]  -= lr * grad_b[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """回傳分類機率（sigmoid 輸出），形狀 (N, 1)。"""
        return self.forward(X, training=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳 0/1 預測。
        - 門檻固定 0.5；若需調整可改為傳入 threshold 參數。
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(float)

    def get_latent_output(self) -> np.ndarray | None:
        """取得在 forward 過程保存的 latent 表徵（倒數第二層 ReLU 輸出）。
        若網路層數不足或尚未 forward，可能回傳 None。
        """
        return self._latent_output

    def get_parameters(self) -> Dict[str, List[np.ndarray]]:
        """取出目前參數（方便存檔/複製/早停快照）。"""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases":  [b.copy() for b in self.biases],
            "layer_sizes": self.layer_sizes,
        }

    def set_parameters(self, params: Dict[str, List[np.ndarray]]) -> None:
        """載入一組參數（需與當前結構相容）。"""
        self.layer_sizes = params["layer_sizes"]
        self.weights = [w.copy() for w in params["weights"]]
        self.biases  = [b.copy() for b in params["biases"]]

# ---------------------------------------------------------------------------
# 損失函數、評估指標與梯度檢查

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """計算平均二元交叉熵（Binary Cross-Entropy, BCE）。

    定義
    ----
    L = - mean( y*log(p) + (1-y)*log(1-p) )
    - y_true ∈ {0,1}，shape = (N,1) 或 (N,)
    - y_pred ∈ (0,1)，通常為 sigmoid 輸出

    數值穩定
    --------
    - 避免 log(0)：先以 eps 夾住機率 p∈[eps, 1-eps]
    - eps 預設 1e-10；若使用 float32，可視需要放寬為 1e-7
    """
    # 防止 log(0) 與 log(1-0) 的溢位
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # 按定義計算 batch 平均 BCE，回傳 Python float
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(loss)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """計算分類常用指標：Loss / Accuracy / Error rate。

    輸入
    ----
    - y_true: 0/1 標籤（(N,1) 或 (N,)）
    - y_pred: 機率（(N,1) 或 (N,)），通常為 sigmoid 輸出

    邏輯
    ----
    - Loss: 使用上面的 BCE
    - Accuracy: 以 0.5 為門檻將機率轉 0/1，再與 y_true 比對
    - Error rate = 1 - Accuracy
    """
    loss = binary_cross_entropy(y_true, y_pred)
    # 機率 → 類別（門檻 0.5；若要調整可外部提供 threshold）
    predictions = (y_pred >= 0.5).astype(float)
    accuracy = float((predictions == y_true).mean())
    error_rate = 1.0 - accuracy
    return {
        "loss": loss,
        "accuracy": accuracy,
        "error_rate": error_rate,
    }


def gradient_check(model: NeuralNetworkClassifier, epsilon: float = 1e-5, sample_size: int = 25) -> float:
    """以中央差分（finite differences）檢查 Backprop 梯度正確性。

    目的
    ----
    - 將解析梯度（backward 計算）與數值梯度（有限差分近似）比較
    - 回傳「最大相對誤差」；門檻常用：
      float64: <1e-6 通常通過；float32: <1e-4

    方法
    ----
    1) 產生小批隨機資料 X 與 0/1 標籤 y
    2) forward(training=True) 後執行 backward 取得解析梯度
    3) 隨機抽取若干權重/偏置元素 θ
       - f(θ+ε) 與 f(θ-ε) 以 **相同**資料前向計算 BCE
       - 數值梯度 ≈ (f(θ+ε) - f(θ-ε)) / (2ε)
       - 相對誤差 = |g_ana - g_num| / max(|g_ana|, |g_num|)
    4) 取所有抽樣中的最大相對誤差回傳

    注意
    ----
    - plus/minus 計算時用 training=False，避免覆寫 cache
    - ReLU 在 0 點不可導，若 z≈0 誤差可能偏大；多抽幾個點較穩
    - 若訓練含正規化，這裡的 f 必須也把正規化項加進去
    """
    rng = np.random.default_rng(CONFIG["seed"])

    # 小批測試資料；輸入維度取自模型第一層大小
    X = rng.normal(size=(8, model.layer_sizes[0]))
    # 二元標籤，shape (8,1)
    y = rng.integers(0, 2, size=(8, 1)).astype(float)

    # 解析梯度：須先 forward(training=True) 建立 cache
    preds = model.forward(X, training=True)
    grad_w_analytic, grad_b_analytic = model.backward(y, preds)

    # 隨機抽樣若干權重元素位置 (layer_idx, i, j)
    weight_positions = []
    for layer_idx, W in enumerate(model.weights):
        rows, cols = W.shape
        for _ in range(min(sample_size, rows * cols)):
            i = rng.integers(0, rows)
            j = rng.integers(0, cols)
            weight_positions.append((layer_idx, i, j))

    # 隨機抽樣若干偏置元素位置 (layer_idx, j)
    bias_positions = []
    for layer_idx, b in enumerate(model.biases):
        cols = b.shape[1]  # b shape = (1, fan_out)
        for _ in range(min(sample_size, cols)):
            j = rng.integers(0, cols)
            bias_positions.append((layer_idx, j))

    max_error = 0.0  # 累積觀測到的最大相對誤差

    # ---- 權重的數值梯度檢查 ----
    for layer_idx, i, j in weight_positions:
        W = model.weights[layer_idx]
        original = W[i, j]  # 保存原值以便還原

        # f(θ+ε)
        W[i, j] = original + epsilon
        plus = binary_cross_entropy(y, model.forward(X, training=False))

        # f(θ-ε)
        W[i, j] = original - epsilon
        minus = binary_cross_entropy(y, model.forward(X, training=False))

        # 還原
        W[i, j] = original

        # 數值梯度（中央差分）
        numerical = (plus - minus) / (2 * epsilon)
        # 對應的解析梯度（backward 結果）
        analytic = grad_w_analytic[layer_idx][i, j]

        # 兩者皆 0 時，跳過（相對誤差無意義）
        if analytic == 0 and numerical == 0:
            continue

        # 相對誤差：對尺度不敏感，並避免除以 0
        relative = abs(analytic - numerical) / max(abs(analytic), abs(numerical))
        max_error = max(max_error, relative)

    # ---- 偏置的數值梯度檢查 ----
    for layer_idx, j in bias_positions:
        b = model.biases[layer_idx]
        original = b[0, j]

        # f(b+ε)
        b[0, j] = original + epsilon
        plus = binary_cross_entropy(y, model.forward(X, training=False))

        # f(b-ε)
        b[0, j] = original - epsilon
        minus = binary_cross_entropy(y, model.forward(X, training=False))

        # 還原
        b[0, j] = original

        numerical = (plus - minus) / (2 * epsilon)
        analytic = grad_b_analytic[layer_idx][0, j]

        if analytic == 0 and numerical == 0:
            continue

        relative = abs(analytic - numerical) / max(abs(analytic), abs(numerical))
        max_error = max(max_error, relative)

    # 回傳最大相對誤差（越小越好）
    return max_error

# ---------------------------------------------------------------------------
# 其他輔助函式：前處理、模型初始化、視覺化等

def preprocess_data(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """完整的資料前處理入口：建立目錄 → 讀檔 → 切分 → 標準化。

    回傳
    ----
    features : {"train": X_tr, "val": X_val, "test": X_te}，各為 np.ndarray (N_split, D)
    targets  : {"train": y_tr, "val": y_val, "test": y_te}，各為 np.ndarray (N_split, 1)
    metadata : 包含 scaler 參數與欄名等中介資訊（之後重建標準化所需）
    """
    paths = config["paths"]                       # 取出所有路徑設定（由上層 CONFIG 管理）

    # 確保所有輸出與中介資料夾存在；不存在就遞迴建立
    ensure_directories(
        [
            paths["processed_dir"],               # 存放處理後 X/y、scaler 參數等 .npy/.json
            paths["artifacts_dir"],               # 訓練產物（最佳權重、歷史、特徵子集結果）
            paths["figures_dir"],                 # 圖檔（學習曲線、散點、重要度、PCA 可視化）
            paths["results_dir"],                 # 統計摘要（metrics JSON）
            paths["latent_dir"],                  # 需要保存 latent 向量或其降維結果時使用
        ]
    )

    df = load_dataset(paths["dataset"])           # 讀取原始資料為 DataFrame（需自行保證欄位一致）

    # 以固定種子做資料切分（避免洩漏，並確保重現性）
    splits = split_dataset(df, seed=config["seed"])  # 典型比例：64/16/20 或你實作中定義的比例

    # 僅以訓練集統計量做 z-score 標準化，並轉為 NumPy；回傳同時包含 metadata
    features, targets, metadata = standardize_features(splits)
    return features, targets, metadata


def initialize_model(input_dim: int, latent_size: int) -> NeuralNetworkClassifier:
    """建立固定拓樸的分類模型。

    結構
    ----
    input_dim → 64 → 32 → latent_size → 1
    - 隱藏層使用 ReLU；輸出層使用 sigmoid（對應二元分類機率）。
    - 權重初始化在 NeuralNetworkClassifier._init_parameters 中用 Glorot uniform。
    """
    layer_sizes = [input_dim, 64, 32, latent_size, 1]                # 指定每層神經元數
    model = NeuralNetworkClassifier(layer_sizes=layer_sizes,         # 建立模型實例
                                    seed=CONFIG["seed"])             # 固定種子以利重現
    return model


def capture_latent(model: NeuralNetworkClassifier, X: np.ndarray) -> np.ndarray:
    """執行一次 forward 並擷取倒數第二層（latent）輸出。

    說明
    ----
    - forward/predict_proba 會在模型內把倒數第二層的激活存到 _latent_output。
    - 若網路層數不足或尚未 forward，_latent_output 可能為 None。
    - 回傳時 copy() 以避免外部修改內部快取。
    """
    _ = model.predict_proba(X)                       # 前向推論以填入 _latent_output（training=False）
    latent = model.get_latent_output()               # 取得最新一次 forward 的 latent 表徵，shape = (N, latent_size)
    if latent is None:
        raise RuntimeError("未能擷取 latent 向量。")  # 典型原因：模型還沒 forward 或結構不含倒數第二層
    return latent.copy()


def pca_reduce(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """使用 SVD 手刻 PCA，降維到 n_components（預設 2 維）方便視覺化。

    步驟
    ----
    1) 中心化：每欄減掉欄均值（zero-mean）。
    2) SVD：centered = U Σ V^T，其中 V 的行是主成分方向。
    3) 取前 n_components 個主方向（從 V^T 的前幾列取得）。
    4) 投影：centered @ components^T → 得到降維座標，shape = (N, n_components)。
    """
    centered = data - data.mean(axis=0, keepdims=True)      # (N, D) → 去均值
    _, _, vh = np.linalg.svd(centered, full_matrices=False) # vh 形狀 (D, D')，其前幾列是主成分的轉置
    components = vh[:n_components]                          # 取前 k 個主方向（行向量的轉置）
    return centered @ components.T                          # 投影到主成分空間 (N, k)


def plot_learning_curves(history: List[TrainingRecord], path: Path) -> None:
    """同時繪製 Loss 與 Accuracy 曲線，利於觀察收斂與過擬合。

    需求假設
    --------
    TrainingRecord 需至少包含欄位：
      - epoch, train_loss, val_loss, train_accuracy, val_accuracy
    """

    # 從歷史紀錄拆出橫軸與四條曲線
    epochs     = [rec.epoch          for rec in history]
    train_loss = [rec.train_loss     for rec in history]
    val_loss   = [rec.val_loss       for rec in history]
    train_acc  = [rec.train_accuracy for rec in history]
    val_acc    = [rec.val_accuracy   for rec in history]

    # 一張圖上疊兩個 y 軸：左 y 軸畫 Loss，右 y 軸畫 Accuracy
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左軸：訓練/驗證 Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 右軸：訓練/驗證 Accuracy（使用 twin y 以避免量級衝突）
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, label="Train Acc", color="#2ca02c", linestyle="--")
    ax2.plot(epochs, val_acc,   label="Val Acc",   color="#d62728", linestyle="--")
    ax2.set_ylabel("Accuracy")

    # 合併兩個座標軸的圖例項
    lines,  labels  = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")

    # 版面與輸出
    plt.title("Classification Learning Curves")
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """手刻 2x2 混淆矩陣圖，顯示 TP/FP/FN/TN 與數量。

    定義（此圖中行=真值、列=預測）：
      row0=True1, row1=True0；col0=Pred1, col1=Pred0
      因此矩陣排列為：
        [[TP, FN],
         [FP, TN]]
    """

    # 計數（注意：y_true/y_pred 允許形狀 (N,1) 或 (N,)）
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    matrix = np.array([[tp, fn],
                       [fp, tn]])
    labels = [["TP", "FN"],
              ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")  # 以顏色深淺顯示數量

    # 在每個格子中央標出類型與數值
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}",
                    ha="center", va="center", color="black", fontsize=12)

    # 設定座標軸刻度與標籤（列=預測，行=真值）
    ax.set_xticks([0, 1], labels=["Pred 1", "Pred 0"])
    ax.set_yticks([0, 1], labels=["True 1", "True 0"])

    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)   # 右側色條作為值大小參考
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_latent_distribution(latent: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    """對 latent 特徵視覺化：>2 維時以 PCA 降到 2D 再散點著色。

    參數
    ----
    latent : (N, K) 來自倒數第二層的表徵
    labels : (N, 1) 或 (N,) 的 0/1 標籤，用於上色
    """

    # 若維度大於 2，先用 PCA 降到 2 維；否則直接使用
    if latent.shape[1] > 2:
        reduced = pca_reduce(latent, n_components=2)
    else:
        reduced = latent

    plt.figure(figsize=(5, 5))
    # 以標籤上色，觀察類別是否在 latent 空間中分離
    plt.scatter(reduced[:, 0], reduced[:, 1],
                c=labels.reshape(-1), cmap="coolwarm", alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.colorbar(label="Label (1=good, 0=bad)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_training_history(history: List[TrainingRecord], path: Path) -> None:
    """把訓練歷史（每個 epoch 一筆）存成 CSV，便於追蹤與重現。"""
    df = pd.DataFrame([asdict(rec) for rec in history])  # dataclass → dict → DataFrame
    df.to_csv(path, index=False)


def evaluate_model(
    model: NeuralNetworkClassifier,
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    history: List[TrainingRecord],
    history_path: Path,
    learning_curve_path: Path,
) -> Dict[str, Dict[str, float]]:
    """統一完成：保存訓練歷史 → 畫學習曲線 → 在三個切分上推論與計分 → 畫混淆矩陣。

    回傳
    ----
    results: {
      "train": {"loss":..., "accuracy":..., "error_rate":...},
      "val":   {...},
      "test":  {...}
    }
    """

    # 1) 存訓練歷史與畫曲線（Loss/Accuracy 同圖雙 y 軸）
    save_training_history(history, history_path)
    plot_learning_curves(history, learning_curve_path)

    # 2) 在 train/val/test 各自推論並計算指標（使用機率輸出）
    results: Dict[str, Dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        probs = model.predict_proba(features[split])
        results[split] = compute_metrics(targets[split], probs)

    # 3) 額外：在測試集繪製混淆矩陣（先轉成 0/1 預測）
    test_preds = (model.predict_proba(features["test"]) >= 0.5).astype(float)
    plot_confusion_matrix(
        targets["test"],
        test_preds,
        CONFIG["paths"]["figures_dir"] / "classification_confusion_matrix.png",
    )
    return results


# ---------------------------------------------------------------------------
# 訓練主迴圈：含早停、梯度裁剪、latent 擷取
# ---------------------------------------------------------------------------

def train_model(
    model: NeuralNetworkClassifier,
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    latent_size: int,
    config: Dict[str, Any],
    store_latent: bool = True,
    perform_gradient_check: bool = True,
) -> Tuple[List[TrainingRecord], Dict[str, Any], Dict[str, np.ndarray]]:
    """核心訓練流程：SGD + 梯度裁剪 + 早停 +（可選）梯度檢查與 latent 追蹤。

    參數
    ----
    model         : 已建立好的分類網路（結構固定，權重隨機初始化）
    features      : {"train": X_tr, "val": X_val, "test": X_te}，各 shape = (N_split, D)
    targets       : {"train": y_tr, "val": y_val, "test": y_te}，各 shape = (N_split, 1)，取值 0/1
    latent_size   : 倒數第二層維度（僅用於進度條顯示）
    config        : 全域設定，需含 "hyperparameters" 子字典（lr/epochs/batch/clip/patience 等）
    store_latent  : 是否在訓練過程擷取幾個時間點的 latent 表徵
    perform_gradient_check : 是否在訓練前做有限差分的梯度檢查（耗時）

    回傳
    ----
    history         : List[TrainingRecord]，每個 epoch 的訓練/驗證指標
    best_state      : dict，get_parameters() 的快照（最佳驗證 Loss 對應的權重/偏置）
    latent_snapshots: dict，key=時間點，value=在驗證集前向後的 latent（如有啟用）
    """

    # —— 讀超參數（學習率/迭代數/批大小/裁剪閾值/早停等） ——
    hyper = config["hyperparameters"]
    lr = hyper["learning_rate"]
    epochs = hyper["epochs"]
    batch_size = hyper["mini_batch_size"]
    clip_value = hyper["gradient_clip_value"]                 # 若為 None 則不裁剪
    patience = hyper["early_stopping_patience"]               # 連續無改善達此值即停止
    min_delta = hyper.get("early_stopping_min_delta", 0.0)    # 最小改善幅度門檻

    # —— 資料切分（僅用 train/val；test 留待最終評估） ——
    train_X, train_y = features["train"], targets["train"]
    val_X,   val_y   = features["val"],   targets["val"]

    # ——（可選）梯度檢查：以有限差分對比 backward 的解析梯度 ——
    if perform_gradient_check:
        rel_error = gradient_check(model)
        if rel_error > 1e-3:
            # 典型容忍：float64 可設更嚴格 <1e-6；此處取 1e-3 作教學容忍
            raise RuntimeError(f"梯度檢查未通過，最大相對誤差 {rel_error:.6f} > 1e-3")

    # —— 訓練歷史與最佳模型狀態的追蹤器 ——
    history: List[TrainingRecord] = []
    best_state = model.get_parameters()           # 先以初始權重作為暫存最佳
    best_val_loss = float("inf")
    epochs_since_best = 0                         # 用於早停計數

    # ——（可選）latent 追蹤：保存若干時間點的驗證集 latent 表徵 ——
    latent_snapshots: Dict[str, np.ndarray] = {}
    best_key: str | None = None
    if store_latent:
        # epoch0_init：訓練前（隨機初始權重）在 val_X 上的 latent 分佈
        latent_snapshots["epoch0_init"] = capture_latent(model, val_X)

    # —— 進度條（顯示 epoch 與當前 lr/損失等） ——
    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc=f"訓練分類模型 (latent={latent_size})",
        unit="epoch",
        leave=False,
    )
    midpoint_epoch = math.ceil(epochs / 2)        # 中點，用於記錄中途 latent

    # ======================== 主訓練回圈 ========================
    for epoch in epoch_iter:
        # —— 每個 epoch 重新洗牌：seed + epoch，確保每輪 batch 次序不同 ——
        for X_batch, y_batch in build_batches(train_X, train_y, batch_size, seed=config["seed"] + epoch):
            # 前向：取得當前批的預測（training=True 以便 backward 使用 cache）
            preds = model.forward(X_batch, training=True)
            # 反向：依 BCE + sigmoid 的組合計算各層梯度
            grad_w, grad_b = model.backward(y_batch, preds)

            # （可選）梯度裁剪：限制梯度元素到 [-clip, clip]，抑制爆炸梯度
            if clip_value is not None:
                grad_w = [np.clip(g, -clip_value, clip_value) for g in grad_w]
                grad_b = [np.clip(g, -clip_value, clip_value) for g in grad_b]

            # 參數更新：最簡 SGD，W -= lr * dW；b 同理
            model.apply_gradients(grad_w, grad_b, lr)

        # —— epoch 結束後：在整個 train/val 上評分（使用 predict_proba，training=False） ——
        train_probs = model.predict_proba(train_X)
        val_probs   = model.predict_proba(val_X)
        train_metrics = compute_metrics(train_y, train_probs)
        val_metrics   = compute_metrics(val_y,   val_probs)

        # 保存當前 epoch 的訓練歷史
        record = TrainingRecord(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_accuracy=train_metrics["accuracy"],
            train_error_rate=train_metrics["error_rate"],
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            val_error_rate=val_metrics["error_rate"],
        )
        history.append(record)

        # 進度條尾註：顯示本輪的 train/val 損失（簡短）
        epoch_iter.set_postfix(
            train_loss=f"{train_metrics['loss']:.3f}",
            val_loss=f"{val_metrics['loss']:.3f}",
        )

        # ——（可選）在特定時間點快照 latent：觀察表示學習的演進 ——
        if store_latent and epoch == 1:
            latent_snapshots["epoch1_after_first_update"] = capture_latent(model, val_X)
        if store_latent and epoch == midpoint_epoch:
            latent_snapshots[f"epoch{epoch}_mid"] = capture_latent(model, val_X)

        # —— 早停邏輯：只要 val loss 有「足夠改善」就刷新最佳；否則累計無改善次數 ——
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = model.get_parameters()    # 快照目前最佳參數
            epochs_since_best = 0

            # 最佳的 latent 只保留最新一次的版本（避免佔空間）
            if store_latent:
                if best_key is not None:
                    latent_snapshots.pop(best_key, None)
                best_key = f"epoch{epoch}_best"
                latent_snapshots[best_key] = capture_latent(model, val_X)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                # 連續 patience 輪沒有達到 min_delta 的改善 → 觸發早停
                break

    epoch_iter.close()

    # —— 還原為最佳驗證表現時的權重狀態（而非最後一輪） ——
    model.set_parameters(best_state)

    # 回傳：完整歷史、最佳狀態快照、以及（如有）latent 快照
    return history, best_state, latent_snapshots



# ---------------------------------------------------------------------------
# Latent 分析與主流程控制
# ---------------------------------------------------------------------------

def run_latent_analysis(
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> None:
    """針對多個 latent 維度重訓模型並紀錄表現與表徵。

    流程
    ----
    1) 逐一取出 config["latent_layer_sizes"] 的候選維度（如 2/4/8/16）。
    2) 針對每個 latent_size：
       - 以相同資料與超參數重訓模型，確保唯一變因是「latent 維度」。
       - 保存訓練歷史與學習曲線。
       - 在 train/val/test 上評估並收集 error rate。
       - 輸出幾個時間點的 latent 向量快照與對應散點圖。
    3) 彙整各 latent_size 的結果為一張 CSV，便於比較。
    """

    latent_dir = config["paths"]["latent_dir"]
    ensure_directories([latent_dir])  # 建立存放 latent 向量快照的根資料夾
    results = []                      # 用來彙整不同 latent_size 的指標

    # 逐一測試不同的 latent 維度
    for latent_size in config["latent_layer_sizes"]:
        tqdm.write(f"[INFO] 重新訓練 latent={latent_size} 的模型以執行比較…")

        # 以相同輸入維度與本輪 latent_size 建立新模型
        model = initialize_model(features["train"].shape[1], latent_size)

        # 訓練：啟用 latent 紀錄；梯度檢查關閉以節省時間
        history, state, snapshots = train_model(
            model,
            features,
            targets,
            latent_size,
            config,
            store_latent=True,
            perform_gradient_check=False,
        )
        # 還原為最佳驗證表現時的權重（train_model 已回傳 best_state）
        model.set_parameters(state)

        # 保存學習曲線與訓練歷史
        history_path = config["paths"]["artifacts_dir"] / f"training_history_latent_{latent_size}.csv"
        learning_curve_path = config["paths"]["figures_dir"] / f"classification_learning_curve_latent_{latent_size}.png"

        # 在三個切分上評估並回傳指標字典
        metrics = evaluate_model(
            model,
            features,
            targets,
            history,
            history_path,
            learning_curve_path,
        )

        # 彙整當前 latent_size 的核心指標（採 error rate 比較直觀）
        results.append(
            {
                "latent_size": latent_size,
                "train_error_rate": metrics["train"]["error_rate"],
                "val_error_rate":   metrics["val"]["error_rate"],
                "test_error_rate":  metrics["test"]["error_rate"],
            }
        )

        # 輸出本輪 latent 向量快照與對應的 2D 視覺化
        latent_subdir = latent_dir / f"latent_{latent_size}"
        ensure_directories([latent_subdir])
        for stage, latent in snapshots.items():
            # .npy 保存原始 latent 數值，便於後續再分析
            np.save(latent_subdir / f"{stage}.npy", latent)
            # 以驗證集標籤上色觀察分佈是否分離；若 latent>2 維，內部會先做 PCA 到 2D
            plot_latent_distribution(
                latent,
                targets["val"],
                title=f"Latent={latent_size} {stage}",
                path=config["paths"]["figures_dir"] / f"classification_latent_{latent_size}_{stage}.png",
            )

    # 將不同 latent_size 的比較結果統整成一張表
    df = pd.DataFrame(results)
    df.to_csv(config["paths"]["latent_comparison"], index=False)






# ---------------------------------------------------------------------------
# 主流程：固定執行完整 full pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """固定執行完整流程：前處理 → 訓練 → 評估 → latent 分析。

    執行結構
    --------
    1) 前處理：建目錄、讀檔、切分、標準化 → 回傳 (features, targets, metadata)
    2) 訓練：以一個 baseline 的 latent 大小訓練模型，保存最佳權重與學習歷史
    3) 評估：在 train/val/test 上計分，並輸出學習曲線與混淆矩陣
    4) latent 分析：針對多個 latent 維度重訓並比較 error rate 與表徵分佈
    """

    # —— 固定隨機性以利重現；準備路徑與輸出資料夾 ——
    set_global_seed(CONFIG["seed"])
    paths = CONFIG["paths"]
    ensure_directories([
        paths["processed_dir"],   # 處理後 X/y、scaler 參數等
        paths["artifacts_dir"],   # 模型產物（最佳權重、歷史、子集合結果）
        paths["figures_dir"],     # 圖檔（學習曲線、混淆矩陣、latent 散點）
        paths["results_dir"],     # 結果 JSON（metrics 摘要）
        paths["latent_dir"],      # 保存 latent 向量快照（.npy）
    ])

    # —— 外層進度條（4 個 stage 固定順序） ——
    stages = ["preprocess", "train", "evaluate", "latent_analysis"]
    stage_bar = tqdm(total=len(stages), desc="流程總進度", unit="stage", leave=False)

    # ======================== Stage 1: 前處理 ========================
    tqdm.write("[INFO] Stage 1/4 - 開始資料前處理…")
    # 內含：建立必要目錄 → 讀取原始資料 → 切分 train/val/test → 僅以 train 的統計做標準化
    features, targets, metadata = preprocess_data(CONFIG)
    stage_bar.update(1)

    # 保守檢查：理論上 preprocess_data 必定回傳 dict；若 None 代表流程不一致
    if metadata is None:
        raise RuntimeError("前處理後 metadata 不應為 None。")

    # ======================== Stage 2: 訓練 ========================
    # 選 baseline latent 維度：若清單 >=2，取第二個作為更「中庸」的基線；否則取第一個
    baseline_latent = (
        CONFIG["latent_layer_sizes"][1]
        if len(CONFIG["latent_layer_sizes"]) > 1
        else CONFIG["latent_layer_sizes"][0]
    )
    tqdm.write(f"[INFO] Stage 2/4 - 訓練 baseline 模型 (latent={baseline_latent})…")

    # 依據訓練特徵維度（欄數）與 baseline latent 建立模型
    model = initialize_model(features["train"].shape[1], baseline_latent)

    # 訓練：啟用 latent 快照與梯度檢查（注意：梯檢較耗時，可視需要關閉）
    history, best_state, snapshots = train_model(
        model,
        features,
        targets,
        baseline_latent,
        CONFIG,
        store_latent=True,
        perform_gradient_check=True,
    )

    # 還原模型到最佳驗證表現時的參數（非最後一個 epoch）
    model.set_parameters(best_state)

    # 持久化：最佳權重（pickle 存 get_parameters 的快照）、訓練歷史與學習曲線
    with paths["best_model"].open("wb") as fp:
        pickle.dump(best_state, fp)
    save_training_history(history, paths["training_history"])
    plot_learning_curves(history, paths["figures_dir"] / "classification_learning_curve.png")

    # 保存本次訓練過程中擷取的 latent 向量與對應可視化
    latent_subdir = paths["latent_dir"] / f"latent_{baseline_latent}_baseline"
    ensure_directories([latent_subdir])
    for stage, latent in snapshots.items():
        # .npy：原始 latent 數值，便於後續再分析或重畫
        np.save(latent_subdir / f"{stage}.npy", latent)
        # 圖：以驗證集標籤上色；若 latent 維度>2，內部會先做 PCA 再畫 2D 散點
        plot_latent_distribution(
            latent,
            targets["val"],
            title=f"Baseline latent={baseline_latent} {stage}",
            path=paths["figures_dir"] / f"classification_latent_baseline_{stage}.png",
        )
    stage_bar.update(1)

    # ======================== Stage 3: 評估 ========================
    tqdm.write("[INFO] Stage 3/4 - 進行模型評估與產出報告…")
    # 直接沿用訓練時收集的歷史（避免重跑）
    history_records = history
    metrics = evaluate_model(
        model,
        features,
        targets,
        history_records,
        paths["training_history"],
        paths["figures_dir"] / "classification_learning_curve.png",
    )
    # 保存三個切分的 metrics 摘要（loss/accuracy/error_rate）
    with paths["summary"].open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    stage_bar.update(1)

    # ======================== Stage 4: latent 分析 ========================
    # 針對 config["latent_layer_sizes"] 的其他候選維度逐一重訓並比較
    tqdm.write("[INFO] Stage 4/4 - 重新訓練不同 latent 維度以比較表示能力…")
    run_latent_analysis(features, targets, CONFIG)
    stage_bar.update(1)

    # 關閉外層進度條
    stage_bar.close()


if __name__ == "__main__":
    # 直接固定執行 full 流程（不提供命令列 phase）
    run_pipeline()

