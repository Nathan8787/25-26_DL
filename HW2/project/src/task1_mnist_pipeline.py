"""
HW2 題組一 MNIST 管線腳本

本腳本嚴格依據 task1_mnist_spec.md 規範實作，透過單一檔案整合
資料前處理、模型訓練、評估視覺化、實驗掃描與結果彙整。
所有關鍵步驟均搭配繁體中文註解，方便後續維護與課程驗證。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import yaml
except ImportError:  # pragma: no cover - 依規格允許 YAML 覆寫，若缺少模組則提示
    yaml = None  # type: ignore

# 為了在無 GUI 環境輸出圖表，強制使用 Agg 後端
plt.switch_backend("Agg")

# 取得專案根目錄
BASE_DIR = Path(__file__).resolve().parent

# 全域設定常數，提供 CLI 或外部檔案覆寫
CONFIG: Dict[str, Any] = {
    "seed": 20250318,
    "paths": {
        "data_root": str(BASE_DIR / "data" / "mnist"),
        "cache_dir": str(BASE_DIR / "data" / "mnist" / "cache"),
        "raw_dir": str(BASE_DIR / "data" / "mnist" / "raw"),
        "artifacts_dir": str(BASE_DIR / "artifacts" / "task1"),
        "checkpoint_dir": str(BASE_DIR / "artifacts" / "task1" / "checkpoints"),
        "figures_dir": str(BASE_DIR / "figures" / "task1"),
        "logs_dir": str(BASE_DIR / "logs" / "task1"),
        "tensorboard_dir": str(Path("logs") / "task1" / "tensorboard"),
        "reports_dir": str(BASE_DIR / "reports" / "task1"),
        "metrics_csv": str(BASE_DIR / "reports" / "task1" / "metrics_collection.csv"),
        "summary_json": str(BASE_DIR / "reports" / "task1" / "summary.json"),
        "durations_json": str(BASE_DIR / "artifacts" / "task1" / "training_durations.json"),
        "metadata_json": str(BASE_DIR / "artifacts" / "task1" / "mnist_metadata.json"),
        "sample_json": str(BASE_DIR / "artifacts" / "task1" / "sample_inspection.json"),
        "feature_map_notes": str(BASE_DIR / "reports" / "task1" / "feature_map_observations.md"),
        "l2_results": str(BASE_DIR / "reports" / "task1" / "l2_results.csv"),
        "stride_filter_results": str(BASE_DIR / "reports" / "task1" / "stride_filter_summary.csv"),
    },
    "base_hyperparameters": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-7,
        "lr_patience": 3,
        "lr_factor": 0.5,
        "early_stop_patience": 7,
    },
    "architecture": {
        "conv_blocks": 3,
        "filters": [32, 64, 128],
        "kernel_sizes": [3, 3, 3],
        "strides": [1, 1, 1],
        "use_batchnorm": True,
        "dropout_rate": 0.5,
    },
    "experiment_grids": {
        "stride_options": [[1, 1, 1], [1, 1, 2], [2, 1, 1]],
        "kernel_options": [[3, 3, 3], [5, 3, 3], [5, 5, 3]],
        "l2_lambdas": [0.0, 1e-5, 1e-4, 5e-4, 1e-3],
    },
    "visualization": {
        "num_correct": 10,
        "num_incorrect": 10,
        "feature_map_depths": [0, 1, 2],
        "feature_map_channels": 16,
    },
    "progress": {
        "use_rich": True,
        "transient": False,
        "bar_refresh_rate": 0.1,
        "stages": [
            "prepare_environment",
            "preprocess_dataset",
            "build_model",
            "train_and_log",
            "evaluate_and_visualize",
            "run_stride_filter_experiments",
            "run_l2_study",
            "summarize_results",
        ],
    },
    "cli_overrides": {
        "mode": ["baseline", "stride_filter", "l2_study", "all"],
        "config": "可選外部 YAML 或 JSON 設定檔路徑",
        "device": ["gpu"],
    },
}

# Rich 主控台，統一輸出格式
console = Console()


def deep_update_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """使用遞迴策略合併巢狀字典，維持 spec 可覆寫設定的需求。"""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config_override(path: str) -> Dict[str, Any]:
    """讀取 YAML 或 JSON 檔案覆寫 CONFIG，並回傳合併後的設定。"""
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"找不到指定的設定檔：{override_path}")
    data: Dict[str, Any]
    if override_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("未安裝 PyYAML，無法讀取 YAML 設定檔。請先執行 pip install pyyaml。")
        with override_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        with override_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    return deep_update_dict(CONFIG.copy(), data)


def ensure_directories(config: Dict[str, Any]) -> None:
    """建立規格要求的資料夾結構，避免後續輸出失敗。"""
    path_values = [
        "cache_dir",
        "raw_dir",
        "artifacts_dir",
        "checkpoint_dir",
        "figures_dir",
        "logs_dir",
        "tensorboard_dir",
        "reports_dir",
    ]
    for key in path_values:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """同步設定 Python、NumPy 與 TensorFlow 的隨機種子以確保可重現。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_device(device: str) -> None:
    """根據 CLI 選項配置 TensorFlow 的運算裝置。"""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            console.print("[yellow]未偵測到 GPU，改用 CPU 執行。[/yellow]")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_active_device_label() -> str:
    """回傳目前計畫使用的運算裝置描述字串。"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU ({len(gpus)} card(s))"
    return "CPU"


def save_json(data: Dict[str, Any], path: Path) -> None:
    """將資料儲存為 JSON，確保使用 UTF-8 編碼。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """將資料列追加至 CSV，若檔案不存在則建立表頭。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False, encoding="utf-8")


def tensorboard_compatible_path(path: Path) -> str:
    """在 Windows 上嘗試轉換成 8.3 短路徑以避免非 ASCII 造成的限制。"""
    if os.name != "nt":
        return str(path)
    try:
        import ctypes

        buffer_len = 1024
        buffer = ctypes.create_unicode_buffer(buffer_len)
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        result = GetShortPathNameW(str(path), buffer, buffer_len)
        if result > 0:
            return buffer.value
    except Exception:
        pass
    return str(path)


def build_progress(config: Dict[str, Any], disable: bool = False) -> Optional["PipelineProgress"]:
    """依設定建立進度控制器。"""
    if disable or not config["progress"]["use_rich"]:
        return None
    return PipelineProgress(config)


class PipelineProgress:
    """封裝 rich Progress 物件，提供階段與子任務的進度管理。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        refresh = max(config["progress"].get("bar_refresh_rate", 0.1), 0.05)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=config["progress"].get("transient", False),
            refresh_per_second=int(1 / refresh),
        )
        self.stage_tasks: Dict[str, TaskID] = {}

    def __enter__(self) -> "PipelineProgress":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.__exit__(exc_type, exc, tb)

    def start_stage(self, name: str, total: int = 1, description: Optional[str] = None) -> None:
        """啟動指定階段的進度追蹤。"""
        desc = description or name
        task_id = self.progress.add_task(desc, total=total)
        self.stage_tasks[name] = task_id

    def advance_stage(self, name: str, advance: int = 1) -> None:
        """更新階段的進度。"""
        task_id = self.stage_tasks.get(name)
        if task_id is not None:
            self.progress.advance(task_id, advance=advance)

    def end_stage(self, name: str) -> None:
        """將指定階段標記為完成。"""
        task_id = self.stage_tasks.get(name)
        if task_id is not None:
            total = self.progress.tasks[task_id].total or 0
            self.progress.update(task_id, completed=total)

    def start_subtask(self, stage: str, title: str, total: int) -> TaskID:
        """在指定階段底下建立子任務。"""
        return self.progress.add_task(f"{stage}:{title}", total=total)

    def advance_subtask(self, task_id: TaskID, advance: int = 1) -> None:
        """更新子任務進度。"""
        self.progress.advance(task_id, advance=advance)

    def end_subtask(self, task_id: TaskID) -> None:
        """結束子任務並補滿進度。"""
        self.progress.update(task_id, completed=self.progress.tasks[task_id].total or 0)


def load_and_split_dataset(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """下載 MNIST 並切分為訓練/驗證/測試資料，同時蒐集統計資訊。"""
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    rng = np.random.default_rng(config["seed"])
    indices = np.arange(x_train_full.shape[0])
    rng.shuffle(indices)

    train_count = 55000
    val_count = 5000
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    num_classes = 10
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

    metadata = {
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(x_test.shape[0]),
        "shape": list(x_train.shape[1:]),
        "class_names": [str(i) for i in range(num_classes)],
    }
    save_json(metadata, Path(config["paths"]["metadata_json"]))

    datasets = {
        "train": (x_train, y_train, y_train_oh),
        "val": (x_val, y_val, y_val_oh),
        "test": (x_test, y_test, y_test_oh),
    }
    return datasets, metadata


def build_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: Dict[str, Any],
) -> Dict[str, tf.data.Dataset]:
    """依照規格建立 tf.data pipeline，包含快取、隨機與預取。"""
    batch_size = config["base_hyperparameters"]["batch_size"]

    def make_dataset(images: np.ndarray, labels_oh: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels_oh))
        if training:
            ds = ds.cache().shuffle(10000, seed=config["seed"])
        else:
            ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_images, _, train_onehot = datasets["train"]
    val_images, _, val_onehot = datasets["val"]
    test_images, _, test_onehot = datasets["test"]

    return {
        "train": make_dataset(train_images, train_onehot, True),
        "val": make_dataset(val_images, val_onehot, False),
        "test": make_dataset(test_images, test_onehot, False),
    }


def build_model(config: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
    """建構可調整卷積區塊的 Keras 模型。"""
    arch = config["architecture"].copy()
    if override:
        for key, value in override.items():
            if key in arch:
                arch[key] = value

    filters = arch["filters"]
    kernels = arch["kernel_sizes"]
    strides = arch["strides"]
    dropout_rate = arch.get("dropout_rate", 0.5)
    use_bn = arch.get("use_batchnorm", True)

    l2_lambda = override.get("l2") if override else None
    regularizer = (
        tf.keras.regularizers.L2(l2_lambda) if l2_lambda is not None and l2_lambda > 0 else None
    )

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for idx, (flt, kern, stride) in enumerate(zip(filters, kernels, strides)):
        x = tf.keras.layers.Conv2D(
            filters=flt,
            kernel_size=kern,
            strides=stride,
            padding="same",
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=config["seed"]),
            kernel_regularizer=regularizer,
            name=f"conv_block{idx+1}_conv1",
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f"conv_block{idx+1}_bn1")(x)
        x = tf.keras.layers.Conv2D(
            filters=flt,
            kernel_size=kern,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=config["seed"]),
            kernel_regularizer=regularizer,
            name=f"conv_block{idx+1}_conv2",
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f"conv_block{idx+1}_bn2")(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"conv_block{idx+1}_pool")(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    if not isinstance(x, tf.Tensor):
        x = tf.keras.layers.Flatten(name="flatten")(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=config["seed"]),
        kernel_regularizer=regularizer,
        name="dense_1",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")


def compile_model(model: tf.keras.Model, config: Dict[str, Any]) -> None:
    """依設定編譯模型，採用 Adam 與交叉熵。"""
    params = config["base_hyperparameters"]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"],
        beta_1=params["beta1"],
        beta_2=params["beta2"],
        epsilon=params["epsilon"],
    )
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


class RichProgressCallback(tf.keras.callbacks.Callback):
    """搭配 rich 進度條顯示 epoch 進度，讓 CLI stage 與訓練同步。"""

    def __init__(self, progress: PipelineProgress, stage_name: str, total_epochs: int):
        super().__init__()
        self.progress = progress
        self.stage_name = stage_name
        self.total_epochs = total_epochs
        self.progress.start_stage(stage_name, total_epochs, description=f"{stage_name} (epoch)")

    def on_epoch_end(self, epoch, logs=None):
        self.progress.advance_stage(self.stage_name, 1)

    def on_train_end(self, logs=None):
        self.progress.end_stage(self.stage_name)


def train_and_log(
    model: tf.keras.Model,
    datasets: Dict[str, tf.data.Dataset],
    config: Dict[str, Any],
    tag: str,
    progress: Optional[PipelineProgress],
) -> Tuple[tf.keras.callbacks.History, Path]:
    """執行訓練並輸出檔案、TensorBoard 日誌與訓練時間統計。"""
    params = config["base_hyperparameters"]
    callbacks: List[tf.keras.callbacks.Callback] = []

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_model_{tag}.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        )
    )
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=params["early_stop_patience"],
            restore_best_weights=True,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=params["lr_factor"],
            patience=params["lr_patience"],
            min_lr=1e-5,
        )
    )
    history_path = Path(config["paths"]["logs_dir"]) / f"history_{tag}.csv"
    callbacks.append(tf.keras.callbacks.CSVLogger(str(history_path)))

    if progress is not None:
        callbacks.append(RichProgressCallback(progress, "train_and_log", params["epochs"]))

    start_time = time.time()
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=params["epochs"],
        callbacks=callbacks,
        verbose=0,
    )
    duration = time.time() - start_time

    durations_path = Path(config["paths"]["durations_json"])
    durations: Dict[str, Any] = {}
    if durations_path.exists():
        with durations_path.open("r", encoding="utf-8") as fh:
            durations = json.load(fh)
    durations[tag] = {
        "seconds": duration,
        "minutes": duration / 60.0,
        "epochs_ran": len(history.history.get("loss", [])),
    }
    save_json(durations, durations_path)

    return history, checkpoint_path


def evaluate_and_visualize(
    model: tf.keras.Model,
    datasets: Dict[str, tf.data.Dataset],
    raw_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    history: tf.keras.callbacks.History,
    config: Dict[str, Any],
    tag: str,
    progress: Optional[PipelineProgress],
) -> Dict[str, Any]:
    """計算指標並輸出規格要求的各式圖表。"""
    if progress is not None:
        progress.start_stage("evaluate_and_visualize", total=1)

    figures_dir = Path(config["paths"]["figures_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = model.evaluate(datasets["train"], verbose=0)
    val_metrics = model.evaluate(datasets["val"], verbose=0)
    test_metrics = model.evaluate(datasets["test"], verbose=0)

    metrics_summary = {
        "train_loss": float(train_metrics[0]),
        "train_accuracy": float(train_metrics[1]),
        "val_loss": float(val_metrics[0]),
        "val_accuracy": float(val_metrics[1]),
        "test_loss": float(test_metrics[0]),
        "test_accuracy": float(test_metrics[1]),
    }

    history_df = pd.DataFrame(history.history)
    history_path = Path(config["paths"]["logs_dir"]) / f"history_{tag}.csv"
    if history_path.exists():
        history_df.to_csv(history_path, index=False, encoding="utf-8")

    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(history_df.index) + 1)
    plt.plot(epochs_range, history_df["loss"], label="train_loss")
    plt.plot(epochs_range, history_df["val_loss"], label="val_loss")
    plt.plot(epochs_range, history_df["accuracy"], label="train_acc")
    plt.plot(epochs_range, history_df["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"learning_curve_{tag}.png", dpi=300)
    plt.close()

    test_images, test_labels, test_onehot = raw_data["test"]
    predictions = model.predict(datasets["test"], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    metrics_summary.update(
        {
            "test_accuracy_macro": float(accuracy_score(test_labels, predicted_labels)),
            "test_precision_macro": float(
                precision_score(test_labels, predicted_labels, average="macro", zero_division=0)
            ),
            "test_recall_macro": float(
                recall_score(test_labels, predicted_labels, average="macro", zero_division=0)
            ),
            "test_f1_macro": float(
                f1_score(test_labels, predicted_labels, average="macro", zero_division=0)
            ),
        }
    )

    cm = confusion_matrix(test_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_{tag}.png", dpi=300)
    plt.close()

    report = classification_report(
        test_labels,
        predicted_labels,
        target_names=[str(i) for i in range(10)],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        reports_dir / f"classification_report_{tag}.csv", encoding="utf-8"
    )

    visualize_correct_incorrect(
        test_images,
        test_labels,
        predictions,
        config,
        tag,
    )
    plot_weight_distributions(model, figures_dir, tag)
    visualize_feature_maps(model, test_images, test_labels, config, tag)

    if progress is not None:
        progress.end_stage("evaluate_and_visualize")

    return metrics_summary


def visualize_correct_incorrect(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    config: Dict[str, Any],
    tag: str,
) -> None:
    """繪製正確與錯誤樣本的比較圖，並記錄詳細資訊。"""
    num_correct = config["visualization"]["num_correct"]
    num_incorrect = config["visualization"]["num_incorrect"]

    predicted_labels = np.argmax(predictions, axis=1)
    confidences = predictions.max(axis=1)

    correct_indices = np.where(predicted_labels == labels)[0]
    incorrect_indices = np.where(predicted_labels != labels)[0]

    correct_order = correct_indices[np.argsort(-confidences[correct_indices])][:num_correct]
    incorrect_order = incorrect_indices[np.argsort(-confidences[incorrect_indices])][:num_incorrect]

    summary = {"correct_samples": [], "incorrect_samples": []}
    plt.figure(figsize=(20, 4))
    for idx, sample_idx in enumerate(correct_order):
        plt.subplot(2, num_correct, idx + 1)
        plt.imshow(images[sample_idx].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(
            f"pred={predicted_labels[sample_idx]}\nlabel={labels[sample_idx]}\nconf={confidences[sample_idx]:.2f}"
        )
        summary["correct_samples"].append(
            {
                "index": int(sample_idx),
                "pred": int(predicted_labels[sample_idx]),
                "label": int(labels[sample_idx]),
                "confidence": float(confidences[sample_idx]),
            }
        )

    for idx, sample_idx in enumerate(incorrect_order):
        plt.subplot(2, num_correct, num_correct + idx + 1)
        plt.imshow(images[sample_idx].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(
            f"pred={predicted_labels[sample_idx]}\nlabel={labels[sample_idx]}\nconf={confidences[sample_idx]:.2f}"
        )
        summary["incorrect_samples"].append(
            {
                "index": int(sample_idx),
                "pred": int(predicted_labels[sample_idx]),
                "label": int(labels[sample_idx]),
                "confidence": float(confidences[sample_idx]),
            }
        )

    plt.suptitle("Correct vs Incorrect Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(config["paths"]["figures_dir"]) / f"correct_vs_incorrect_{tag}.png", dpi=300)
    plt.close()

    save_json(summary, Path(config["paths"]["sample_json"]))


def plot_weight_distributions(model: tf.keras.Model, figures_dir: Path, tag: str) -> None:
    """輸出各層權重與偏差的分佈圖，滿足規格中的統計需求。"""
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        for idx, weight_array in enumerate(weights):
            flattened = weight_array.flatten()
            plt.figure(figsize=(6, 4))
            sns.histplot(flattened, kde=True, bins=30)
            plt.title(f"Layer {layer.name} - Param {idx}")
            plt.tight_layout()
            plt.savefig(figures_dir / f"weights_{layer.name}_{idx}_{tag}.png", dpi=300)
            plt.close()


def visualize_feature_maps(
    model: tf.keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config: Dict[str, Any],
    tag: str,
) -> None:
    """擷取卷積層特徵圖並輸出，另生成文字敘述檔供報告使用。"""
    conv_layers = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=conv_layers)

    representatives = []
    for label in range(5):
        indices = np.where(test_labels == label)[0]
        if len(indices) == 0:
            continue
        representatives.append(indices[0])

    feature_map_notes: List[str] = []
    channels_to_show = config["visualization"]["feature_map_channels"]

    for sample_idx in representatives:
        sample_input = test_images[sample_idx : sample_idx + 1]
        feature_maps = intermediate_model.predict(sample_input, verbose=0)
        for layer_idx, fmap in enumerate(feature_maps):
            if fmap.ndim != 4:
                continue
            fmap = fmap[0]
            total_channels = fmap.shape[-1]
            num_channels = min(channels_to_show, total_channels)
            cols = max(1, int(math.sqrt(num_channels)))
            rows = math.ceil(num_channels / cols)
            plt.figure(figsize=(cols * 2, rows * 2))
            for channel_idx in range(num_channels):
                plt.subplot(rows, cols, channel_idx + 1)
                plt.imshow(fmap[:, :, channel_idx], cmap="viridis")
                plt.axis("off")
            plt.suptitle(
                f"Sample {sample_idx} (label={test_labels[sample_idx]}) - Layer {layer_idx}",
                fontsize=12,
            )
            plt.tight_layout()
            plt.savefig(
                Path(config["paths"]["figures_dir"])
                / f"feature_maps_sample{sample_idx}_layer{layer_idx}_{tag}.png",
                dpi=300,
            )
            plt.close()
            feature_map_notes.append(
                f"樣本 {sample_idx} 在第 {layer_idx} 層有 {total_channels} 個特徵圖，觀察前 {num_channels} 個通道可見由邊緣到筆畫的抽象變化。"
            )

    notes_path = Path(config["paths"]["feature_map_notes"])
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(
        "\n".join(feature_map_notes) if feature_map_notes else "尚無特徵圖資訊。",
        encoding="utf-8",
    )


def run_stride_filter_experiments(
    datasets_raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    progress: Optional[PipelineProgress],
) -> None:
    """逐一測試不同 stride 與 kernel 組合，並記錄指標。"""
    stride_options = config["experiment_grids"]["stride_options"]
    kernel_options = config["experiment_grids"]["kernel_options"]
    combos = list(product(stride_options, kernel_options))

    if progress is not None:
        progress.start_stage("run_stride_filter_experiments", total=len(combos))

    datasets = build_datasets(datasets_raw, config)
    for strides, kernels in combos:
        tag = f"stride{'-'.join(map(str, strides))}_kernel{'-'.join(map(str, kernels))}"
        override = {"strides": list(strides), "kernel_sizes": list(kernels)}
        model = build_model(config, override=override)
        compile_model(model, config)
        history, _ = train_and_log(model, datasets, config, tag, progress)
        metrics = evaluate_and_visualize(model, datasets, datasets_raw, history, config, tag, progress)
        metrics.update({"tag": tag, "strides": list(strides), "kernels": list(kernels)})
        append_csv_row(Path(config["paths"]["stride_filter_results"]), metrics)
        if progress is not None:
            progress.advance_stage("run_stride_filter_experiments", 1)

    if progress is not None:
        progress.end_stage("run_stride_filter_experiments")


def run_l2_study(
    datasets: Dict[str, tf.data.Dataset],
    datasets_raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    progress: Optional[PipelineProgress],
) -> None:
    """掃描 L2 正則化係數並記錄結果。"""
    lambdas = config["experiment_grids"]["l2_lambdas"]
    if progress is not None:
        progress.start_stage("run_l2_study", total=len(lambdas))

    for l2_value in lambdas:
        tag = f"l2_{l2_value:.0e}".replace("+", "")
        override = {"l2": l2_value}
        model = build_model(config, override=override)
        compile_model(model, config)
        history, _ = train_and_log(model, datasets, config, tag, progress)
        metrics = evaluate_and_visualize(model, datasets, datasets_raw, history, config, tag, progress)
        weight_norm = float(
            sum(np.linalg.norm(weight) for layer in model.layers for weight in layer.get_weights())
        )
        metrics.update({"lambda": l2_value, "weight_norm": weight_norm, "tag": tag})
        append_csv_row(Path(config["paths"]["l2_results"]), metrics)
        if progress is not None:
            progress.advance_stage("run_l2_study", 1)

    if progress is not None:
        progress.end_stage("run_l2_study")


def summarize_results(baseline_metrics: Dict[str, Any], config: Dict[str, Any], tag: str = "baseline") -> None:
    """彙整主要指標並輸出 summary.json。"""
    summary_path = Path(config["paths"]["summary_json"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {"best_config": {"tag": tag}, "metrics": baseline_metrics}
    save_json(summary_data, summary_path)


def run_sanity_checks(datasets: Dict[str, tf.data.Dataset], model: tf.keras.Model) -> None:
    """快速驗證資料形狀與模型輸出是否符合預期。"""
    images, labels = next(iter(datasets["train"]))
    assert images.shape[-1] == 1, "影像應為單通道。"
    outputs = model(images[:2])
    assert outputs.shape[-1] == 10, "模型輸出需對應 10 個類別。"


def run_pipeline(args: argparse.Namespace) -> None:
    """依據 CLI 模式執行整體管線。"""
    config = CONFIG.copy()
    if args.config:
        config = load_config_override(args.config)
    if args.no_progress:
        config["progress"]["use_rich"] = False

    ensure_directories(config)
    configure_device(args.device)
    set_seed(config["seed"])

    datasets_raw, metadata = load_and_split_dataset(config)
    datasets = build_datasets(datasets_raw, config)

    progress_manager = build_progress(config, disable=not config["progress"]["use_rich"])
    baseline_metrics: Dict[str, Any] = {}

    with (progress_manager or NullContext()) as progress:
        device_label = get_active_device_label()
        console.print(f"[bold blue]使用裝置: {device_label}[/bold blue]")
        if progress is not None:
            progress.start_stage("prepare_environment")
            progress.end_stage("prepare_environment")
            progress.start_stage("preprocess_dataset")
            progress.end_stage("preprocess_dataset")

        if args.mode in {"baseline", "all"}:
            if progress is not None:
                progress.start_stage("build_model")
            baseline_model = build_model(config)
            compile_model(baseline_model, config)
            if progress is not None:
                progress.end_stage("build_model")

            run_sanity_checks(datasets, baseline_model)
            history, _ = train_and_log(baseline_model, datasets, config, "baseline", progress)
            baseline_metrics = evaluate_and_visualize(
                baseline_model,
                datasets,
                datasets_raw,
                history,
                config,
                "baseline",
                progress,
            )
            summarize_results(baseline_metrics, config, tag="baseline")

        if args.mode in {"stride_filter", "all"}:
            run_stride_filter_experiments(datasets_raw, config, progress)

        if args.mode in {"l2_study", "all"}:
            run_l2_study(datasets, datasets_raw, config, progress)

        if progress is not None and args.mode in {"baseline", "all"}:
            progress.start_stage("summarize_results")
            progress.end_stage("summarize_results")

    console.print("[bold green]MNIST Pipeline 執行完畢。[/bold green]")


class NullContext:
    """簡易的 nullcontext，確保無進度條時也能使用 with 敘述。"""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """解析 CLI 參數並回傳命名空間。"""
    parser = argparse.ArgumentParser(description="HW2 MNIST Pipeline")
    parser.add_argument(
        "--mode",
        choices=CONFIG["cli_overrides"]["mode"],
        default="baseline",
        help="指定要執行的流程段落",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="外部 YAML/JSON 覆寫設定檔路徑",
    )
    parser.add_argument(
        "--device",
        choices=CONFIG["cli_overrides"]["device"],
        default="auto",
        help="選擇運算裝置",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="停用 rich 進度條與狀態欄",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """程式進入點：解析參數並啟動管線。"""
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main(sys.argv[1:])
