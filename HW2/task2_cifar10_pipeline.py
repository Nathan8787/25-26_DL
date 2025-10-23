"""
HW2 題組二 CIFAR-10 腳本

這份腳本是為了滿足 HW2 第二題的需求，
1. 資料下載、前處理與簡單增強
2. baseline 模型訓練與評估
3. stride / kernel 比較
4. L2 正則化實驗
5. 基本的前處理消融說明
6. 視覺化正確/錯誤樣本與卷積特徵圖
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import albumentations as A
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
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

plt.switch_backend("Agg")

BASE_DIR = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    "seed": 20250318,
    "paths": {
        "data_root": str(BASE_DIR / "data" / "cifar10"),
        "cache_dir": str(BASE_DIR / "data" / "cifar10" / "cache"),
        "raw_dir": str(BASE_DIR / "data" / "cifar10" / "raw"),
        "artifacts_dir": str(BASE_DIR / "artifacts" / "task2"),
        "checkpoint_dir": str(BASE_DIR / "artifacts" / "task2" / "checkpoints"),
        "figures_dir": str(BASE_DIR / "figures" / "task2"),
        "logs_dir": str(BASE_DIR / "logs" / "task2"),
        "reports_dir": str(BASE_DIR / "reports" / "task2"),
        "metadata_json": str(BASE_DIR / "artifacts" / "task2" / "cifar10_metadata.json"),
        "class_names": str(BASE_DIR / "artifacts" / "task2" / "class_names.json"),
        "channel_stats": str(BASE_DIR / "artifacts" / "task2" / "channel_stats.json"),
        "feature_map_notes": str(BASE_DIR / "reports" / "task2" / "feature_map_observations.md"),
        "preprocessing_md": str(BASE_DIR / "reports" / "task2" / "preprocessing.md"),
        "stride_filter_results": str(BASE_DIR / "reports" / "task2" / "stride_filter_results.csv"),
        "l2_results": str(BASE_DIR / "reports" / "task2" / "l2_results.csv"),
        "preprocessing_ablation": str(BASE_DIR / "reports" / "task2" / "preprocessing_ablation.csv"),
        "summary_json": str(BASE_DIR / "reports" / "task2" / "summary.json"),
        "durations_json": str(BASE_DIR / "artifacts" / "task2" / "training_durations.json"),
    },
    "base_hyperparameters": {
        "epochs": 120,
        "batch_size": 256,
        "learning_rate": 2e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "cosine_first_decay": 20,
        "cosine_t_mul": 2.0,
        "cosine_m_mul": 0.9,
        "early_stop_patience": 12,
    },
    "augmentation": {
        "random_flip": True,
        "random_crop": True,
        "crop_padding": 4,
        "random_rotation_deg": 15,
        "random_zoom": 0.1,
        "random_translation": 0.1,
        "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02},
        "cutout": {"size": 8, "prob": 0.3},
        "mixup_alpha": 0.2,
        "use_mixup": False,
    },
    "architecture": {
        "filters": [64, 128, 256, 512],
        "kernels": [3, 3, 3, 3],
        "strides": [1, 1, 1, 1],
        "dropout_rate": 0.5,
        "use_batchnorm": True,
    },
    "experiment_grids": {
        "stride_options": [[1, 1, 1, 1], [1, 1, 2, 1], [2, 1, 1, 1]],
        "kernel_options": [[3, 3, 3, 3], [5, 3, 3, 3], [5, 5, 3, 3]],
        "l2_lambdas": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "preprocessing_variants": [
            "baseline",
            "no_standardization",
            "no_augmentation",
            "no_cutout",
            "no_color",
            "no_mixup",
        ],
    },
    "visualization": {
        "correct_samples": 15,
        "incorrect_samples": 15,
        "feature_map_blocks": [1, 2, 3],
        "feature_map_channels": 16,
    },
    "progress": {
        "use_rich": True,
        "transient": False,
        "refresh_rate": 0.1,
        "stages": [
            "prepare_environment",
            "preprocess_dataset",
            "build_model",
            "train_and_log",
            "evaluate_visualize",
            "run_stride_filter_grid",
            "run_l2_regularization_study",
            "run_preprocessing_ablation",
            "summarize_results",
        ],
    },
    "cli_overrides": {
        "mode": ["baseline", "stride_filter", "l2_study", "preprocessing_ablation", "all"],
        "config": "外部覆寫設定檔路徑",
        "device": ["cpu", "gpu", "auto"],
        "mixed_precision": [True, False],
    },
}

console = Console()


def deep_update_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """遞迴合併巢狀字典，確保 CONFIG 可被外部設定覆寫。"""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config_override(path: str) -> Dict[str, Any]:
    """讀取 YAML 或 JSON 檔並覆寫預設 CONFIG。"""
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"找不到設定檔：{override_path}")
    if override_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("需要 PyYAML 才能讀取 YAML 設定檔。")
        with override_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        with override_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    return deep_update_dict(CONFIG.copy(), data)


def ensure_directories(config: Dict[str, Any]) -> None:
    """建立規格要求的資料夾結構。"""
    keys = [
        "cache_dir",
        "raw_dir",
        "artifacts_dir",
        "checkpoint_dir",
        "figures_dir",
        "logs_dir",
        "reports_dir",
    ]
    for key in keys:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """設定隨機種子，確保實驗可重現。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_device(device: str, mixed_precision: bool) -> None:
    """按照使用者選項配置運算裝置與混合精度策略。"""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            console.print("[yellow]未偵測到 GPU，將改用 CPU。[/yellow]")
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

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def save_json(data: Dict[str, Any], path: Path) -> None:
    """儲存 JSON 檔案，採用 UTF-8 與美化縮排。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    """追加資料到 CSV，若不存在則建立新檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False, encoding="utf-8")


def build_progress(config: Dict[str, Any], disable: bool = False) -> Optional["PipelineProgress"]:
    """依設定建立進度條控制器。"""
    if disable or not config["progress"]["use_rich"]:
        return None
    return PipelineProgress(config)


class PipelineProgress:
    """封裝 rich Progress，以階段方式呈現執行狀態。"""

    def __init__(self, config: Dict[str, Any]):
        refresh_rate = max(config["progress"].get("refresh_rate", 0.1), 0.05)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=config["progress"].get("transient", False),
            refresh_per_second=int(1 / refresh_rate),
        )
        self.tasks: Dict[str, TaskID] = {}

    def __enter__(self) -> "PipelineProgress":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.progress.__exit__(exc_type, exc, tb)

    def start_stage(self, name: str, total: int = 1, description: Optional[str] = None) -> None:
        task_id = self.progress.add_task(description or name, total=total)
        self.tasks[name] = task_id

    def advance_stage(self, name: str, step: int = 1) -> None:
        task_id = self.tasks.get(name)
        if task_id is not None:
            self.progress.advance(task_id, step)

    def end_stage(self, name: str) -> None:
        task_id = self.tasks.get(name)
        if task_id is not None:
            total = self.progress.tasks[task_id].total or 0
            self.progress.update(task_id, completed=total)

    def start_subtask(self, stage: str, title: str, total: int) -> TaskID:
        return self.progress.add_task(f"{stage}:{title}", total=total)

    def advance_subtask(self, task_id: TaskID, step: int = 1) -> None:
        self.progress.advance(task_id, step)

    def end_subtask(self, task_id: TaskID) -> None:
        total = self.progress.tasks[task_id].total or 0
        self.progress.update(task_id, completed=total)


def load_and_split_dataset(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """載入 CIFAR-10，並依規格切分成訓練/驗證/測試集合。"""
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    rng = np.random.default_rng(config["seed"])
    val_per_class = 500  # 10 類別 * 500 = 5000
    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls in range(10):
        cls_indices = np.where(y_train_full == cls)[0]
        rng.shuffle(cls_indices)
        val_indices.extend(cls_indices[:val_per_class])
        train_indices.extend(cls_indices[val_per_class:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    save_json({"class_names": class_names}, Path(config["paths"]["class_names"]))

    metadata = {
        "train_count": int(x_train.shape[0]),
        "val_count": int(x_val.shape[0]),
        "test_count": int(x_test.shape[0]),
        "shape": list(x_train.shape[1:]),
        "class_names": class_names,
    }
    save_json(metadata, Path(config["paths"]["metadata_json"]))

    datasets = {
        "train": {"images": x_train, "labels": y_train},
        "val": {"images": x_val, "labels": y_val},
        "test": {"images": x_test, "labels": y_test},
    }
    return datasets, metadata


def calculate_channel_stats(train_images: np.ndarray, config: Dict[str, Any]) -> Dict[str, List[float]]:
    """計算通道均值與標準差，用於標準化。"""
    mean = train_images.mean(axis=(0, 1, 2)).tolist()
    std = train_images.std(axis=(0, 1, 2)).tolist()
    stats = {"mean": mean, "std": std}
    save_json(stats, Path(config["paths"]["channel_stats"]))
    return stats


def build_albumentations_pipeline(aug_settings: Dict[str, Any]) -> Optional[A.Compose]:
    """建立訓練階段使用的資料增強流程。"""
    aug_cfg = aug_settings
    if not aug_cfg.get("enable", True):
        return None
    transforms: List[A.BasicTransform] = []
    if aug_cfg["random_flip"]:
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg["random_crop"]:
        pad = aug_cfg["crop_padding"]
        transforms.append(A.PadIfNeeded(32 + pad, 32 + pad, border_mode=0, value=0))
        transforms.append(A.RandomCrop(32, 32))
    transforms.append(
        A.ShiftScaleRotate(
            shift_limit=aug_cfg["random_translation"],
            scale_limit=aug_cfg["random_zoom"],
            rotate_limit=aug_cfg["random_rotation_deg"],
            border_mode=1,
            value=0,
            p=0.9,
        )
    )
    transforms.append(
        A.ColorJitter(
            brightness=aug_cfg["color_jitter"]["brightness"],
            contrast=aug_cfg["color_jitter"]["contrast"],
            saturation=aug_cfg["color_jitter"]["saturation"],
            hue=aug_cfg["color_jitter"]["hue"],
            p=0.8,
        )
    )
    transforms.append(
        A.CoarseDropout(
            max_holes=1,
            max_height=aug_cfg["cutout"]["size"],
            max_width=aug_cfg["cutout"]["size"],
            fill_value=0,
            p=aug_cfg["cutout"]["prob"],
        )
    )
    return A.Compose(transforms)


def build_datasets(
    datasets: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
    stats: Dict[str, List[float]],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, tf.data.Dataset]:
    """依規格建立 tf.data 管線，加入資料增強與標準化。"""
    batch_size = config["base_hyperparameters"]["batch_size"]
    options = options or {}

    aug_settings = copy.deepcopy(config["augmentation"])
    if not options.get("augmentation", True):
        aug_settings["enable"] = False
    if options.get("no_cutout", False):
        aug_settings["cutout"]["prob"] = 0.0
    if options.get("no_color", False):
        aug_settings["color_jitter"] = {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0}
    aug_settings["use_mixup"] = options.get("mixup", aug_settings["use_mixup"])
    aug_pipeline = build_albumentations_pipeline(aug_settings)

    standardize_enabled = options.get("standardize", True)
    if standardize_enabled:
        mean = tf.constant(stats["mean"], dtype=tf.float32)
        std = tf.constant(stats["std"], dtype=tf.float32)

    def standardize(image: tf.Tensor) -> tf.Tensor:
        if standardize_enabled:
            return (image - mean) / (std + 1e-7)
        return image

    def augment_image(image: np.ndarray) -> np.ndarray:
        if aug_pipeline is None:
            return image.astype(np.float32)
        augmented = aug_pipeline(image=image)["image"]
        return augmented.astype(np.float32)

    def train_map(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(augment_image, [image], tf.float32)
        image.set_shape((32, 32, 3))
        image = standardize(image)
        label_onehot = tf.one_hot(label, 10)
        label_onehot.set_shape((10,))
        return image, label_onehot

    def eval_map(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = standardize(image)
        label_onehot = tf.one_hot(label, 10)
        label_onehot.set_shape((10,))
        return image, label_onehot

    def make_dataset(images: np.ndarray, labels: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(10000, seed=config["seed"])
            ds = ds.map(train_map, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(eval_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if training and aug_settings["use_mixup"]:
            ds = ds.map(lambda x, y: apply_mixup(x, y, aug_settings["mixup_alpha"]), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(datasets["train"]["images"], datasets["train"]["labels"], True)
    val_ds = make_dataset(datasets["val"]["images"], datasets["val"]["labels"], False)
    test_ds = make_dataset(datasets["test"]["images"], datasets["test"]["labels"], False)
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def apply_mixup(images: tf.Tensor, labels: tf.Tensor, alpha: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """在 batch 維度進行 MixUp，增強模型泛化能力。"""
    alpha = tf.cast(alpha, tf.float32)
    gamma1 = tf.random.gamma([], alpha, 1.0)
    gamma2 = tf.random.gamma([], alpha, 1.0)
    lam = gamma1 / (gamma1 + gamma2 + 1e-7)
    lam_x = tf.reshape(lam, [1, 1, 1, 1])
    lam_y = tf.reshape(lam, [1, 1])

    indices = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)

    mixed_images = lam_x * images + (1 - lam_x) * images2
    mixed_labels = lam_y * labels + (1 - lam_y) * labels2
    return mixed_images, mixed_labels


def build_model(config: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
    """依據規格建構可調整的 CNN 模型。"""
    arch = config["architecture"].copy()
    if override:
        for key, value in override.items():
            if key in arch:
                arch[key] = value

    filters = arch["filters"]
    kernels = arch["kernels"]
    strides = arch["strides"]
    dropout_rate = arch["dropout_rate"]
    use_bn = arch["use_batchnorm"]

    l2_lambda = override.get("l2") if override else None
    regularizer = (
        tf.keras.regularizers.L2(l2_lambda) if l2_lambda is not None and l2_lambda > 0 else None
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs

    for idx, (flt, ker, stride) in enumerate(zip(filters, kernels, strides)):
        x = tf.keras.layers.Conv2D(
            filters=flt,
            kernel_size=ker,
            strides=stride,
            padding="same",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=config["seed"]),
            kernel_regularizer=regularizer,
            name=f"block{idx+1}_conv1",
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f"block{idx+1}_bn1")(x)
        x = tf.keras.layers.Conv2D(
            filters=flt,
            kernel_size=ker,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=config["seed"]),
            kernel_regularizer=regularizer,
            name=f"block{idx+1}_conv2",
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f"block{idx+1}_bn2")(x)
        if idx < len(filters) - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"block{idx+1}_pool")(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = tf.keras.layers.Dense(
        512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=config["seed"]),
        kernel_regularizer=regularizer,
        name="dense_1",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """結合線性暖機與 CosineDecayRestarts 的學習率排程。"""

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cosine_schedule: tf.keras.optimizers.schedules.CosineDecayRestarts,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cosine_schedule = cosine_schedule

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = self.base_lr * (step + 1) / self.warmup_steps
            cosine_lr = self.cosine_schedule(tf.maximum(step - self.warmup_steps, 0))
            return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
        return self.cosine_schedule(step)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "cosine_schedule": self.cosine_schedule.get_config(),
        }


def compile_model(
    model: tf.keras.Model,
    config: Dict[str, Any],
    steps_per_epoch: int,
) -> None:
    """編譯模型，採用 AdamW 與含暖機的餘弦衰減。"""
    params = config["base_hyperparameters"]
    warmup_steps = params["warmup_epochs"] * steps_per_epoch
    cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=params["learning_rate"],
        first_decay_steps=params["cosine_first_decay"] * steps_per_epoch,
        t_mul=params["cosine_t_mul"],
        m_mul=params["cosine_m_mul"],
    )
    lr_schedule = WarmupCosineSchedule(params["learning_rate"], warmup_steps, cosine)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=params["weight_decay"],
    )
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "categorical_accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )


class RichProgressCallback(tf.keras.callbacks.Callback):
    """使用 rich 進度條呈現訓練 epoch 進度。"""

    def __init__(self, progress: PipelineProgress, stage: str, total_epochs: int):
        super().__init__()
        self.progress = progress
        self.stage = stage
        self.total_epochs = total_epochs
        self.progress.start_stage(stage, total_epochs, description=f"{stage} (epoch)")

    def on_epoch_end(self, epoch, logs=None):
        self.progress.advance_stage(self.stage, 1)

    def on_train_end(self, logs=None):
        self.progress.end_stage(self.stage)


def train_and_log(
    model: tf.keras.Model,
    datasets: Dict[str, tf.data.Dataset],
    config: Dict[str, Any],
    tag: str,
    progress: Optional[PipelineProgress],
) -> Tuple[tf.keras.callbacks.History, Path]:
    """執行訓練並記錄 checkpoint、CSV log 與訓練時間。"""
    params = config["base_hyperparameters"]
    callbacks: List[tf.keras.callbacks.Callback] = []

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_model_{tag}.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_categorical_accuracy",
            save_best_only=True,
            save_weights_only=False,
        )
    )
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_categorical_accuracy",
            patience=params["early_stop_patience"],
            restore_best_weights=True,
        )
    )

    csv_log_path = Path(config["paths"]["logs_dir"]) / f"history_{tag}.csv"
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks.append(tf.keras.callbacks.CSVLogger(str(csv_log_path)))

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
    durations = {}
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
    raw_data: Dict[str, Dict[str, np.ndarray]],
    history: tf.keras.callbacks.History,
    config: Dict[str, Any],
    tag: str,
    progress: Optional[PipelineProgress],
) -> Dict[str, Any]:
    """計算指標並產出各項圖表與分析資料。"""
    if progress is not None:
        progress.start_stage("evaluate_visualize", total=1)

    figures_dir = Path(config["paths"]["figures_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = model.evaluate(datasets["train"], verbose=0)
    val_metrics = model.evaluate(datasets["val"], verbose=0)
    test_metrics = model.evaluate(datasets["test"], verbose=0)

    metrics_summary = {
        "train_loss": float(train_metrics[0]),
        "train_top1": float(train_metrics[1]),
        "train_top5": float(train_metrics[2]),
        "val_loss": float(val_metrics[0]),
        "val_top1": float(val_metrics[1]),
        "val_top5": float(val_metrics[2]),
        "test_loss": float(test_metrics[0]),
        "test_top1": float(test_metrics[1]),
        "test_top5": float(test_metrics[2]),
    }

    history_df = pd.DataFrame(history.history)
    csv_path = Path(config["paths"]["logs_dir"]) / f"history_{tag}.csv"
    if csv_path.exists():
        history_df.to_csv(csv_path, index=False, encoding="utf-8")

    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(history_df.index) + 1)
    plt.plot(epochs_range, history_df["loss"], label="train_loss")
    plt.plot(epochs_range, history_df["val_loss"], label="val_loss")
    plt.plot(epochs_range, history_df["categorical_accuracy"], label="train_top1")
    plt.plot(epochs_range, history_df["val_categorical_accuracy"], label="val_top1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"learning_curve_{tag}.png", dpi=300)
    plt.close()

    test_images = raw_data["test"]["images"]
    test_labels = raw_data["test"]["labels"]
    predictions = model.predict(datasets["test"], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    top5_preds = np.argsort(predictions, axis=1)[:, -5:]

    top1_acc = accuracy_score(test_labels, pred_labels)
    top5_acc = np.mean([label in top5 for label, top5 in zip(test_labels, top5_preds)])
    precision_macro = precision_score(test_labels, pred_labels, average="macro", zero_division=0)
    recall_macro = recall_score(test_labels, pred_labels, average="macro", zero_division=0)
    f1_macro = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    metrics_summary.update(
        {
            "test_top1_macro": float(top1_acc),
            "test_top5_macro": float(top5_acc),
            "test_precision_macro": float(precision_macro),
            "test_recall_macro": float(recall_macro),
            "test_f1_macro": float(f1_macro),
        }
    )

    cm = confusion_matrix(test_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_{tag}.png", dpi=300)
    plt.close()

    report = classification_report(
        test_labels,
        pred_labels,
        target_names=[str(i) for i in range(10)],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        reports_dir / f"classification_report_{tag}.csv",
        encoding="utf-8",
    )

    visualize_samples(
        test_images,
        test_labels,
        predictions,
        config,
        tag,
    )
    plot_weight_distributions(model, figures_dir, tag)
    visualize_feature_maps(model, test_images, test_labels, config, tag)

    if progress is not None:
        progress.end_stage("evaluate_visualize")

    return metrics_summary


def visualize_samples(
    images: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    config: Dict[str, Any],
    tag: str,
) -> None:
    """將正確與錯誤樣本輸出為圖像，輔助報告分析。"""
    num_correct = config["visualization"]["correct_samples"]
    num_incorrect = config["visualization"]["incorrect_samples"]
    figures_dir = Path(config["paths"]["figures_dir"])

    pred_labels = np.argmax(probabilities, axis=1)
    confidences = probabilities.max(axis=1)

    correct_idx = np.where(pred_labels == labels)[0]
    incorrect_idx = np.where(pred_labels != labels)[0]
    correct_order = correct_idx[np.argsort(-confidences[correct_idx])][:num_correct]
    incorrect_order = incorrect_idx[np.argsort(-confidences[incorrect_idx])][:num_incorrect]

    summary = {"correct_samples": [], "incorrect_samples": []}
    plt.figure(figsize=(20, 6))
    for idx, sample_idx in enumerate(correct_order):
        plt.subplot(2, num_correct, idx + 1)
        plt.imshow(images[sample_idx])
        plt.axis("off")
        plt.title(
            f"pred={pred_labels[sample_idx]}\nlabel={labels[sample_idx]}\nconf={confidences[sample_idx]:.2f}"
        )
        summary["correct_samples"].append(
            {
                "index": int(sample_idx),
                "pred": int(pred_labels[sample_idx]),
                "label": int(labels[sample_idx]),
                "confidence": float(confidences[sample_idx]),
            }
        )

    for idx, sample_idx in enumerate(incorrect_order):
        plt.subplot(2, num_correct, num_correct + idx + 1)
        plt.imshow(images[sample_idx])
        plt.axis("off")
        plt.title(
            f"pred={pred_labels[sample_idx]}\nlabel={labels[sample_idx]}\nconf={confidences[sample_idx]:.2f}"
        )
        summary["incorrect_samples"].append(
            {
                "index": int(sample_idx),
                "pred": int(pred_labels[sample_idx]),
                "label": int(labels[sample_idx]),
                "confidence": float(confidences[sample_idx]),
            }
        )

    plt.suptitle("Correct vs Incorrect Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / f"correct_vs_incorrect_{tag}.png", dpi=300)
    plt.close()

    save_json(summary, Path(config["paths"]["artifacts_dir"]) / f"sample_summary_{tag}.json")


def plot_weight_distributions(model: tf.keras.Model, figures_dir: Path, tag: str) -> None:
    """繪製模型各層權重與偏差的分佈圖。"""
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        for idx, arr in enumerate(weights):
            flattened = arr.flatten()
            plt.figure(figsize=(6, 4))
            sns.histplot(flattened, kde=True, bins=30)
            plt.title(f"Layer {layer.name} Param {idx}")
            plt.tight_layout()
            plt.savefig(figures_dir / f"weights_{layer.name}_{idx}_{tag}.png", dpi=300)
            plt.close()


def visualize_feature_maps(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    tag: str,
) -> None:
    """擷取卷積層特徵圖並輸出圖像與文字描述。"""
    block_indices = config["visualization"]["feature_map_blocks"]
    conv_layers = [
        layer.output
        for idx, layer in enumerate(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=conv_layers)

    notes: List[str] = []
    for cls in range(10):
        indices = np.where(labels == cls)[0]
        if len(indices) == 0:
            continue
        sample_idx = indices[0]
        sample = images[sample_idx : sample_idx + 1]
        feature_maps = intermediate_model.predict(sample, verbose=0)
        for block_idx in block_indices:
            if block_idx >= len(feature_maps):
                continue
            fmap = feature_maps[block_idx][0]
            total_channels = fmap.shape[-1]
            num_channels = min(config["visualization"]["feature_map_channels"], total_channels)
            cols = max(1, int(math.sqrt(num_channels)))
            rows = math.ceil(num_channels / cols)
            plt.figure(figsize=(cols * 2, rows * 2))
            for ch in range(num_channels):
                plt.subplot(rows, cols, ch + 1)
                plt.imshow(fmap[:, :, ch], cmap="viridis")
                plt.axis("off")
            plt.suptitle(f"Class {cls} - Block {block_idx}", fontsize=12)
            plt.tight_layout()
            plt.savefig(
                Path(config["paths"]["figures_dir"]) / f"featuremaps_cls{cls}_block{block_idx}_{tag}.png",
                dpi=300,
            )
            plt.close()
            notes.append(
                f"類別 {cls} 在卷積區塊 {block_idx} 顯示 {total_channels} 個特徵圖，前 {num_channels} 個通道呈現顏色邊緣與紋理層次。"
            )

    notes_path = Path(config["paths"]["feature_map_notes"])
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(notes) if notes else "尚無特徵圖紀錄。", encoding="utf-8")


def run_stride_filter_grid(
    datasets: Dict[str, tf.data.Dataset],
    raw_data: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
    stats: Dict[str, List[float]],
    steps_per_epoch: int,
    progress: Optional[PipelineProgress],
) -> None:
    """根據規格逐一評估不同 stride/filter 組合。"""
    combos = list(product(config["experiment_grids"]["stride_options"], config["experiment_grids"]["kernel_options"]))
    if progress is not None:
        progress.start_stage("run_stride_filter_grid", total=len(combos))

    for strides, kernels in combos:
        tag = f"stride{'-'.join(map(str, strides))}_kernel{'-'.join(map(str, kernels))}"
        override = {"strides": list(strides), "kernels": list(kernels)}
        model = build_model(config, override=override)
        compile_model(model, config, steps_per_epoch)
        history, _ = train_and_log(model, datasets, config, tag, progress)
        metrics = evaluate_and_visualize(model, datasets, raw_data, history, config, tag, progress)
        metrics.update({"tag": tag, "strides": list(strides), "kernels": list(kernels)})
        append_csv(Path(config["paths"]["stride_filter_results"]), metrics)
        if progress is not None:
            progress.advance_stage("run_stride_filter_grid", 1)

    if progress is not None:
        progress.end_stage("run_stride_filter_grid")


def run_l2_regularization_study(
    datasets: Dict[str, tf.data.Dataset],
    raw_data: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
    steps_per_epoch: int,
    progress: Optional[PipelineProgress],
) -> None:
    """測試指定的 L2 正則化係數並比較結果。"""
    lambdas = config["experiment_grids"]["l2_lambdas"]
    if progress is not None:
        progress.start_stage("run_l2_regularization_study", total=len(lambdas))

    for l2_value in lambdas:
        tag = f"l2_{l2_value:.0e}".replace("+", "")
        model = build_model(config, override={"l2": l2_value})
        compile_model(model, config, steps_per_epoch)
        history, _ = train_and_log(model, datasets, config, tag, progress)
        metrics = evaluate_and_visualize(model, datasets, raw_data, history, config, tag, progress)
        weight_norm = float(sum(np.linalg.norm(w) for layer in model.layers for w in layer.get_weights()))
        metrics.update({"lambda": l2_value, "weight_norm": weight_norm, "tag": tag})
        append_csv(Path(config["paths"]["l2_results"]), metrics)
        if progress is not None:
            progress.advance_stage("run_l2_regularization_study", 1)

    if progress is not None:
        progress.end_stage("run_l2_regularization_study")


def run_preprocessing_ablation(
    raw_data: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
    stats: Dict[str, List[float]],
    steps_per_epoch: int,
    progress: Optional[PipelineProgress],
) -> None:
    """針對指定的前處理變體進行消融實驗。"""
    variants = config["experiment_grids"]["preprocessing_variants"]
    if progress is not None:
        progress.start_stage("run_preprocessing_ablation", total=len(variants))

    records = []
    for variant in variants:
        variant_cfg = copy.deepcopy(config)
        options = {"standardize": True, "augmentation": True, "no_cutout": False, "no_color": False, "mixup": variant_cfg["augmentation"]["use_mixup"]}
        if variant == "baseline":
            pass
        elif variant == "no_standardization":
            options["standardize"] = False
        elif variant == "no_augmentation":
            options["augmentation"] = False
        elif variant == "no_cutout":
            options["no_cutout"] = True
        elif variant == "no_color":
            options["no_color"] = True
        elif variant == "no_mixup":
            options["mixup"] = False
        else:
            console.print(f"[yellow]未知前處理變體：{variant}，略過。[/yellow]")
            continue

        datasets_variant = build_datasets(raw_data, variant_cfg, stats, options=options)
        model = build_model(variant_cfg)
        compile_model(model, variant_cfg, steps_per_epoch)
        history, _ = train_and_log(model, datasets_variant, variant_cfg, f"pre_{variant}", progress)
        metrics = evaluate_and_visualize(
            model,
            datasets_variant,
            raw_data,
            history,
            variant_cfg,
            f"pre_{variant}",
            progress,
        )
        metrics.update({"variant": variant})
        records.append(metrics)
        append_csv(Path(config["paths"]["preprocessing_ablation"]), metrics)
        if progress is not None:
            progress.advance_stage("run_preprocessing_ablation", 1)

    write_preprocessing_report(records, Path(config["paths"]["preprocessing_md"]))

    if progress is not None:
        progress.end_stage("run_preprocessing_ablation")


def write_preprocessing_report(records: List[Dict[str, Any]], path: Path) -> None:
    """將前處理消融結果寫入 Markdown，方便報告引用。"""
    lines = ["# Preprocessing Ablation Summary", ""]
    for record in records:
        lines.append(f"## Variant: {record.get('variant', 'unknown')}")
        lines.append(f"- Test Top-1 Accuracy: {record.get('test_top1', 0):.4f}")
        lines.append(f"- Test Top-5 Accuracy: {record.get('test_top5', 0):.4f}")
        lines.append(f"- Macro F1: {record.get('test_f1_macro', 0):.4f}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) if lines else "尚無消融結果。", encoding="utf-8")


def summarize_results(metrics: Dict[str, Any], config: Dict[str, Any], tag: str) -> None:
    """輸出 baseline 結果摘要。"""
    summary = {"best_config": {"tag": tag}, "metrics": metrics}
    save_json(summary, Path(config["paths"]["summary_json"]))


def run_sanity_checks(datasets: Dict[str, tf.data.Dataset]) -> None:
    """執行簡單的 sanity check，確保資料形狀與範圍合理。"""
    sample_images, sample_labels = next(iter(datasets["train"]))
    assert sample_images.shape[-1] == 3, "CIFAR-10 影像必須為 RGB。"
    assert sample_labels.shape[-1] == 10, "標籤應為 10 維 one-hot 編碼。"


def run_pipeline(args: argparse.Namespace) -> None:
    """主控流程：解析設定、建立資料與模型，並依模式執行實驗。"""
    config = CONFIG.copy()
    if args.config:
        config = load_config_override(args.config)
    if args.no_progress:
        config["progress"]["use_rich"] = False

    ensure_directories(config)
    configure_device(args.device, args.mixed_precision)
    set_seed(config["seed"])

    datasets_raw, metadata = load_and_split_dataset(config)
    stats = calculate_channel_stats(datasets_raw["train"]["images"], config)
    datasets = build_datasets(datasets_raw, config, stats)
    steps_per_epoch = math.ceil(len(datasets_raw["train"]["images"]) / config["base_hyperparameters"]["batch_size"])

    progress_manager = build_progress(config, disable=not config["progress"]["use_rich"])
    baseline_metrics: Dict[str, Any] = {}

    with (progress_manager or NullContext()) as progress:
        if progress is not None:
            progress.start_stage("prepare_environment")
            progress.end_stage("prepare_environment")
            progress.start_stage("preprocess_dataset")
            progress.end_stage("preprocess_dataset")

        if args.mode in {"baseline", "all"}:
            if progress is not None:
                progress.start_stage("build_model")
            baseline_model = build_model(config)
            compile_model(baseline_model, config, steps_per_epoch)
            if progress is not None:
                progress.end_stage("build_model")

            run_sanity_checks(datasets)
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
            run_stride_filter_grid(datasets, datasets_raw, config, stats, steps_per_epoch, progress)

        if args.mode in {"l2_study", "all"}:
            run_l2_regularization_study(datasets, datasets_raw, config, steps_per_epoch, progress)

        if args.mode in {"preprocessing_ablation", "all"}:
            run_preprocessing_ablation(datasets_raw, config, stats, steps_per_epoch, progress)

        if progress is not None and args.mode in {"baseline", "all"}:
            progress.start_stage("summarize_results")
            progress.end_stage("summarize_results")

    console.print("[bold green]CIFAR-10 Pipeline 已完成。[/bold green]")


class NullContext:
    """與 with 語句搭配的空上下文管理器。"""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """解析 CLI 參數。"""
    parser = argparse.ArgumentParser(description="HW2 Task2 CIFAR-10 Pipeline")
    parser.add_argument(
        "--mode",
        choices=CONFIG["cli_overrides"]["mode"],
        default="baseline",
        help="選擇要執行的流程段落",
    )
    parser.add_argument("--config", type=str, help="外部 YAML/JSON 設定檔")
    parser.add_argument(
        "--device",
        choices=CONFIG["cli_overrides"]["device"],
        default="auto",
        help="指定運算裝置",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="啟用混合精度訓練",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="停用 rich 進度條輸出",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """程式進入點。"""
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main(sys.argv[1:])
