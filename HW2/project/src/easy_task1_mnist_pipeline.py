"""
簡化版 MNIST 管線
------------------
這支腳本保留作業要求的核心功能，但刻意保持結構單純，方便理解與維護。

功能列表：
1. 基本 CNN 訓練與驗證。
2. 不同 stride / kernel 的小型網格實驗。
3. L2 正則化對準確率的影響。
4. 正確與錯誤樣本視覺化、卷積特徵圖觀察。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# 避免在無 GUI 環境中彈出視窗
plt.switch_backend("Agg")

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainResult:
    """簡單封裝訓練結果，方便儲存與生成報表。"""

    tag: str
    history: tf.keras.callbacks.History
    best_model_path: Path


CONFIG = {
    "seed": 2025,
    "paths": {
        "artifacts": BASE_DIR / "artifacts" / "easy_task1",
        "figures": BASE_DIR / "figures" / "easy_task1",
        "reports": BASE_DIR / "reports" / "easy_task1",
    },
    "training": {
        # epoch 數量故意設少一點，方便迭代；真正調整時再提高即可
        "epochs": 25,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "early_stop_patience": 5,
    },
    "experiments": {
        "stride_options": [1, 2],
        "kernel_options": [3, 5],
        "l2_values": [0.0, 1e-4, 5e-4],
    },
    "visualization": {
        "num_samples": 20,
        "feature_map_layers": ["conv_1", "conv_2"],
    },
}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def ensure_directories(paths: Iterable[Path]) -> None:
    """一次建立所有需要用到的資料夾。"""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """鎖定 Python / NumPy / TensorFlow 的亂數來源。"""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(data: Dict, path: Path) -> None:
    """輸出 JSON，維持 UTF-8 格式與縮排。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 資料處理
# ---------------------------------------------------------------------------
def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """載入 MNIST 並將像素縮放到 [0, 1]。"""

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def split_train_val(
    images: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """手動切 validation；MNIST 原始資料沒有額外的驗證集合。"""

    total = images.shape[0]
    val_count = int(total * val_ratio)
    x_val = images[-val_count:]
    y_val = labels[-val_count:]
    x_train = images[:-val_count]
    y_train = labels[:-val_count]
    return x_train, y_train, x_val, y_val


def make_tf_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> Dict[str, tf.data.Dataset]:
    """建立簡單的 tf.data pipeline。"""

    def _build(images: np.ndarray, labels: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images[..., None], labels))
        if training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)

    return {
        "train": _build(x_train, y_train, training=True),
        "val": _build(x_val, y_val, training=False),
        "test": _build(x_test, y_test, training=False),
    }


# ---------------------------------------------------------------------------
# 模型建構與訓練
# ---------------------------------------------------------------------------
def build_cnn(
    num_filters: int = 32,
    kernel_size: int = 3,
    stride: int = 1,
    l2_value: float = 0.0,
) -> tf.keras.Model:
    """極簡 CNN：兩個卷積區塊 + MLP 頭。"""

    regularizer = tf.keras.regularizers.L2(l2_value) if l2_value > 0 else None

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
        name="conv_1",
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(
        num_filters * 2,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
        name="conv_2",
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_simple_cnn")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """統一的編譯設定。"""

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(
    model: tf.keras.Model,
    datasets: Dict[str, tf.data.Dataset],
    config: Dict,
    tag: str,
) -> TrainResult:
    """回傳訓練歷程與最佳模型路徑。"""

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["training"]["early_stop_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CONFIG["paths"]["artifacts"] / f"best_model_{tag}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    return TrainResult(
        tag=tag,
        history=history,
        best_model_path=CONFIG["paths"]["artifacts"] / f"best_model_{tag}.keras",
    )


# ---------------------------------------------------------------------------
# 報表與視覺化
# ---------------------------------------------------------------------------
def plot_learning_curve(history: tf.keras.callbacks.History, tag: str) -> None:
    """輸出 train/val accuracy 與 loss 的折線圖。"""

    history_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history_df["accuracy"], label="train")
    axes[0].plot(history_df["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history_df["loss"], label="train")
    axes[1].plot(history_df["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.suptitle(f"Learning Curve - {tag}")
    fig.tight_layout()
    output_path = CONFIG["paths"]["figures"] / f"learning_curve_{tag}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    history_csv = CONFIG["paths"]["reports"] / f"history_{tag}.csv"
    history_df.to_csv(history_csv, index=False, encoding="utf-8")


def evaluate_and_report(
    model: tf.keras.Model,
    datasets: Dict[str, tf.data.Dataset],
    tag: str,
) -> None:
    """計算分類指標並輸出混淆矩陣與分類報告。"""

    logits = model.predict(datasets["test"], verbose=0)
    predictions = np.argmax(logits, axis=1)

    y_true = np.concatenate([batch_y.numpy() for _, batch_y in datasets["test"]])

    report = classification_report(y_true, predictions, output_dict=True)
    report_path = CONFIG["paths"]["reports"] / f"classification_report_{tag}.csv"
    pd.DataFrame(report).transpose().to_csv(report_path, encoding="utf-8")

    cm = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix - {tag}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(CONFIG["paths"]["figures"] / f"confusion_matrix_{tag}.png", dpi=200)
    plt.close(fig)


def plot_correct_incorrect(
    model: tf.keras.Model, datasets: Dict[str, tf.data.Dataset], tag: str
) -> None:
    """蒐集預測最有把握的正確 / 錯誤樣本，畫成網格圖。"""

    logits = model.predict(datasets["test"], verbose=0)
    y_true = np.concatenate([batch_y.numpy() for _, batch_y in datasets["test"]])
    y_pred = logits.argmax(axis=1)
    confidences = logits.max(axis=1)

    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    correct_indices = np.argsort(confidences[correct_mask])[-CONFIG["visualization"]["num_samples"] :]
    incorrect_indices = np.argsort(confidences[incorrect_mask])[-CONFIG["visualization"]["num_samples"] :]

    def _plot(indices: np.ndarray, subset: np.ndarray, title: str, filename: str) -> None:
        cols = 5
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()
        for ax, idx in zip(axes, indices):
            img = subset[idx][..., 0]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        for ax in axes[len(indices) :]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(CONFIG["paths"]["figures"] / filename, dpi=200)
        plt.close(fig)

    test_images = np.concatenate([batch_x.numpy() for batch_x, _ in datasets["test"]])
    _plot(correct_indices, test_images[correct_mask], "Correct Samples", f"correct_{tag}.png")
    _plot(incorrect_indices, test_images[incorrect_mask], "Incorrect Samples", f"incorrect_{tag}.png")


def visualize_feature_maps(
    model: tf.keras.Model, datasets: Dict[str, tf.data.Dataset], tag: str
) -> None:
    """拉出測試集中幾張圖，觀察兩個卷積層的激活結果。"""

    conv_layers = [model.get_layer(name) for name in CONFIG["visualization"]["feature_map_layers"]]
    feature_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])

    # 取測試資料前一個 batch 即可
    sample_batch = next(iter(datasets["test"]))[0][:5]
    activations = feature_model(sample_batch, training=False)

    for layer_name, act in zip(CONFIG["visualization"]["feature_map_layers"], activations):
        num_filters = min(act.shape[-1], 8)
        fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
        for idx in range(num_filters):
            axes[idx].imshow(act[0, :, :, idx], cmap="viridis")
            axes[idx].axis("off")
        fig.suptitle(f"{layer_name} feature maps ({tag})")
        fig.tight_layout()
        fig.savefig(CONFIG["paths"]["figures"] / f"feature_map_{layer_name}_{tag}.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 不同作業模式
# ---------------------------------------------------------------------------
def run_baseline(datasets: Dict[str, tf.data.Dataset]) -> tf.keras.Model:
    """訓練 baseline 並輸出所有作業需要的圖表。"""

    model = build_cnn()
    compile_model(model, CONFIG["training"]["learning_rate"])
    result = train_model(model, datasets, CONFIG, tag="baseline")
    plot_learning_curve(result.history, "baseline")
    evaluate_and_report(model, datasets, "baseline")
    plot_correct_incorrect(model, datasets, "baseline")
    visualize_feature_maps(model, datasets, "baseline")
    return model


def run_stride_filter_experiments(datasets: Dict[str, tf.data.Dataset]) -> None:
    """少量組合即可呈現趨勢。"""

    records: List[Dict] = []
    for stride in CONFIG["experiments"]["stride_options"]:
        for kernel in CONFIG["experiments"]["kernel_options"]:
            tag = f"stride{stride}_kernel{kernel}"
            model = build_cnn(num_filters=32, kernel_size=kernel, stride=stride)
            compile_model(model, CONFIG["training"]["learning_rate"])
            result = train_model(model, datasets, CONFIG, tag=tag)
            _, acc = model.evaluate(datasets["test"], verbose=0)
            records.append({"tag": tag, "stride": stride, "kernel": kernel, "test_accuracy": float(acc)})
            plot_learning_curve(result.history, tag)

    report_path = CONFIG["paths"]["reports"] / "stride_kernel_summary.csv"
    pd.DataFrame(records).to_csv(report_path, index=False, encoding="utf-8")


def run_l2_experiments(datasets: Dict[str, tf.data.Dataset]) -> None:
    """比較不同 L2 強度對最終測試表現的影響。"""

    rows: List[Dict] = []
    for l2_value in CONFIG["experiments"]["l2_values"]:
        tag = f"l2_{l2_value:.0e}".replace("+", "")
        model = build_cnn(l2_value=l2_value)
        compile_model(model, CONFIG["training"]["learning_rate"])
        result = train_model(model, datasets, CONFIG, tag=tag)
        loss, acc = model.evaluate(datasets["test"], verbose=0)
        rows.append({"lambda": l2_value, "test_loss": float(loss), "test_accuracy": float(acc)})
        plot_learning_curve(result.history, tag)

    pd.DataFrame(rows).to_csv(CONFIG["paths"]["reports"] / "l2_results.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="簡化版 MNIST 作業管線")
    parser.add_argument(
        "--mode",
        choices=["baseline", "stride_filter", "l2", "all"],
        default="baseline",
        help="選擇要執行的功能區段",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories(CONFIG["paths"].values())
    set_global_seed(CONFIG["seed"])

    x_train, y_train, x_test, y_test = load_mnist()
    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)

    datasets = make_tf_datasets(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=CONFIG["training"]["batch_size"],
    )

    if args.mode in {"baseline", "all"}:
        run_baseline(datasets)
    if args.mode in {"stride_filter", "all"}:
        run_stride_filter_experiments(datasets)
    if args.mode in {"l2", "all"}:
        run_l2_experiments(datasets)


if __name__ == "__main__":
    main()
