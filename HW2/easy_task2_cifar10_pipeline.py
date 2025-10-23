"""
簡化版 CIFAR-10 管線
---------------------
這份腳本是為了滿足 HW2 第二題的基本需求，但去掉繁複的自動化網格與進階功能，
保留最重要的：
1. 資料下載、前處理與簡單增強
2. baseline 模型訓練與評估
3. stride / kernel 比較
4. L2 正則化實驗
5. 基本的前處理消融說明
6. 視覺化正確/錯誤樣本與卷積特徵圖
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

plt.switch_backend("Agg")

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainResult:
    """封裝訓練歷程與最佳模型路徑。"""

    tag: str
    history: tf.keras.callbacks.History
    best_model_path: Path


CONFIG = {
    "seed": 2025,
    "paths": {
        "artifacts": BASE_DIR / "artifacts" / "easy_task2",
        "figures": BASE_DIR / "figures" / "easy_task2",
        "reports": BASE_DIR / "reports" / "easy_task2",
    },
    "training": {
        "epochs": 40,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "early_stop_patience": 6,
    },
    "experiments": {
        "stride_options": [[1, 1, 1], [1, 2, 1]],
        "kernel_options": [[3, 3, 3], [5, 3, 3]],
        "l2_values": [0.0, 1e-4, 5e-4],
        "preprocessing_variants": ["baseline", "no_standardization", "no_augmentation"],
    },
    "augmentation": {
        "random_flip": True,
        "random_crop": True,
        "crop_padding": 4,
    },
    "visualization": {
        "num_samples": 20,
        "feature_map_layers": ["block1_conv", "block2_conv", "block3_conv"],
    },
}


# ---------------------------------------------------------------------------
# 共用函式
# ---------------------------------------------------------------------------
def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 資料與前處理
# ---------------------------------------------------------------------------
def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_train.astype("float32") / 255.0, y_train.flatten(), x_test.astype("float32") / 255.0, y_test.flatten()


def split_train_val(
    images: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_count = int(images.shape[0] * val_ratio)
    return (
        images[:-val_count],
        labels[:-val_count],
        images[-val_count:],
        labels[-val_count:],
    )


def compute_channel_stats(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    std[std == 0.0] = 1.0
    return mean, std


def standardize_images(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (images - mean) / std


def augment_image(image: tf.Tensor) -> tf.Tensor:
    if CONFIG["augmentation"]["random_crop"]:
        image = tf.image.resize_with_crop_or_pad(
            image,
            32 + CONFIG["augmentation"]["crop_padding"],
            32 + CONFIG["augmentation"]["crop_padding"],
        )
        image = tf.image.random_crop(image, size=(32, 32, 3))
    if CONFIG["augmentation"]["random_flip"]:
        image = tf.image.random_flip_left_right(image)
    return image


def make_tf_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    use_standardization: bool,
    use_augmentation: bool,
) -> Dict[str, tf.data.Dataset]:
    mean, std = compute_channel_stats(x_train) if use_standardization else (np.zeros(3), np.ones(3))

    def _preprocess(image: tf.Tensor, label: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        image = (image - mean) / std
        if training and use_augmentation:
            image = augment_image(image)
        return image, label

    def _build(images: np.ndarray, labels: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(10000)
        ds = ds.map(lambda img, lab: _preprocess(img, lab, training), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)

    return {
        "train": _build(x_train, y_train, True),
        "val": _build(x_val, y_val, False),
        "test": _build(x_test, y_test, False),
    }


# ---------------------------------------------------------------------------
# 模型建構與訓練
# ---------------------------------------------------------------------------
def build_cnn(
    filters: List[int],
    kernels: List[int],
    strides: List[int],
    l2_value: float = 0.0,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.L2(l2_value) if l2_value > 0 else None

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs
    for idx, (flt, ker, stride) in enumerate(zip(filters, kernels, strides), start=1):
        x = tf.keras.layers.Conv2D(
            flt,
            kernel_size=ker,
            strides=stride,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
            name=f"block{idx}_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar_simple_cnn")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
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
    return TrainResult(tag=tag, history=history, best_model_path=CONFIG["paths"]["artifacts"] / f"best_model_{tag}.keras")


# ---------------------------------------------------------------------------
# 評估與視覺化
# ---------------------------------------------------------------------------
def plot_learning_curve(history: tf.keras.callbacks.History, tag: str) -> None:
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
    fig.savefig(CONFIG["paths"]["figures"] / f"learning_curve_{tag}.png", dpi=200)
    plt.close(fig)
    history_df.to_csv(CONFIG["paths"]["reports"] / f"history_{tag}.csv", index=False, encoding="utf-8")


def evaluate_and_report(model: tf.keras.Model, datasets: Dict[str, tf.data.Dataset], tag: str) -> None:
    logits = model.predict(datasets["test"], verbose=0)
    predictions = np.argmax(logits, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in datasets["test"]])

    report = classification_report(y_true, predictions, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(CONFIG["paths"]["reports"] / f"classification_report_{tag}.csv", encoding="utf-8")

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


def plot_correct_incorrect(model: tf.keras.Model, datasets: Dict[str, tf.data.Dataset], tag: str) -> None:
    logits = model.predict(datasets["test"], verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in datasets["test"]])
    preds = logits.argmax(axis=1)
    confidences = logits.max(axis=1)

    correct_mask = preds == y_true
    incorrect_mask = ~correct_mask

    correct_indices = np.argsort(confidences[correct_mask])[-CONFIG["visualization"]["num_samples"] :]
    incorrect_indices = np.argsort(confidences[incorrect_mask])[-CONFIG["visualization"]["num_samples"] :]

    test_images = np.concatenate([x.numpy() for x, _ in datasets["test"]])

    def _plot(indices: np.ndarray, subset: np.ndarray, filename: str, title: str) -> None:
        cols = 5
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()
        for ax, idx in zip(axes, indices):
            img = subset[idx]
            ax.imshow((img - img.min()) / (img.max() - img.min() + 1e-6))
            ax.axis("off")
        for ax in axes[len(indices) :]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(CONFIG["paths"]["figures"] / filename, dpi=200)
        plt.close(fig)

    _plot(correct_indices, test_images[correct_mask], f"correct_{tag}.png", "Correct Samples")
    _plot(incorrect_indices, test_images[incorrect_mask], f"incorrect_{tag}.png", "Incorrect Samples")


def visualize_feature_maps(model: tf.keras.Model, datasets: Dict[str, tf.data.Dataset], tag: str) -> None:
    conv_layers = [model.get_layer(name) for name in CONFIG["visualization"]["feature_map_layers"]]
    feature_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])

    sample_batch = next(iter(datasets["test"]))[0][:3]
    activations = feature_model(sample_batch, training=False)

    for layer_name, act in zip(CONFIG["visualization"]["feature_map_layers"], activations):
        num_filters = min(act.shape[-1], 6)
        fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
        for idx in range(num_filters):
            axes[idx].imshow(act[0, :, :, idx], cmap="viridis")
            axes[idx].axis("off")
        fig.suptitle(f"{layer_name} ({tag})")
        fig.tight_layout()
        fig.savefig(CONFIG["paths"]["figures"] / f"feature_maps_{layer_name}_{tag}.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 作業模式
# ---------------------------------------------------------------------------
def run_baseline(datasets: Dict[str, tf.data.Dataset]) -> tf.keras.Model:
    default_filters = [32, 64, 128]
    default_kernels = [3, 3, 3]
    default_strides = [1, 1, 1]

    model = build_cnn(default_filters, default_kernels, default_strides)
    compile_model(model, CONFIG["training"]["learning_rate"])
    result = train_model(model, datasets, CONFIG, tag="baseline")
    plot_learning_curve(result.history, "baseline")
    evaluate_and_report(model, datasets, "baseline")
    plot_correct_incorrect(model, datasets, "baseline")
    visualize_feature_maps(model, datasets, "baseline")
    return model


def run_stride_kernel_experiments(datasets: Dict[str, tf.data.Dataset]) -> None:
    filters = [32, 64, 128]
    records: List[Dict] = []
    for strides in CONFIG["experiments"]["stride_options"]:
        for kernels in CONFIG["experiments"]["kernel_options"]:
            tag = f"stride{'-'.join(map(str, strides))}_kernel{'-'.join(map(str, kernels))}"
            model = build_cnn(filters, kernels, strides)
            compile_model(model, CONFIG["training"]["learning_rate"])
            result = train_model(model, datasets, CONFIG, tag=tag)
            loss, acc = model.evaluate(datasets["test"], verbose=0)
            records.append(
                {
                    "tag": tag,
                    "strides": "-".join(map(str, strides)),
                    "kernels": "-".join(map(str, kernels)),
                    "test_loss": float(loss),
                    "test_accuracy": float(acc),
                }
            )
            plot_learning_curve(result.history, tag)
    pd.DataFrame(records).to_csv(CONFIG["paths"]["reports"] / "stride_kernel_results.csv", index=False, encoding="utf-8")


def run_l2_experiments(datasets: Dict[str, tf.data.Dataset]) -> None:
    filters = [32, 64, 128]
    kernels = [3, 3, 3]
    strides = [1, 1, 1]

    rows: List[Dict] = []
    for l2_value in CONFIG["experiments"]["l2_values"]:
        tag = f"l2_{l2_value:.0e}".replace("+", "")
        model = build_cnn(filters, kernels, strides, l2_value=l2_value)
        compile_model(model, CONFIG["training"]["learning_rate"])
        result = train_model(model, datasets, CONFIG, tag=tag)
        loss, acc = model.evaluate(datasets["test"], verbose=0)
        rows.append({"lambda": l2_value, "test_loss": float(loss), "test_accuracy": float(acc)})
        plot_learning_curve(result.history, tag)
    pd.DataFrame(rows).to_csv(CONFIG["paths"]["reports"] / "l2_results.csv", index=False, encoding="utf-8")


def run_preprocessing_ablation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    rows: List[Dict] = []
    for variant in CONFIG["experiments"]["preprocessing_variants"]:
        use_standardization = variant != "no_standardization"
        use_augmentation = variant != "no_augmentation"

        tag = f"prep_{variant}"
        datasets = make_tf_datasets(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            CONFIG["training"]["batch_size"],
            use_standardization=use_standardization,
            use_augmentation=use_augmentation,
        )

        model = build_cnn([32, 64, 128], [3, 3, 3], [1, 1, 1])
        compile_model(model, CONFIG["training"]["learning_rate"])
        result = train_model(model, datasets, CONFIG, tag=tag)
        loss, acc = model.evaluate(datasets["test"], verbose=0)
        rows.append(
            {
                "variant": variant,
                "standardization": use_standardization,
                "augmentation": use_augmentation,
                "test_loss": float(loss),
                "test_accuracy": float(acc),
            }
        )
        plot_learning_curve(result.history, tag)

    pd.DataFrame(rows).to_csv(CONFIG["paths"]["reports"] / "preprocessing_ablation.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# 命令列介面
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="簡化版 CIFAR-10 管線")
    parser.add_argument(
        "--mode",
        choices=["baseline", "stride_kernel", "l2", "preprocess", "all"],
        default="baseline",
        help="選擇要執行的流程",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories(CONFIG["paths"].values())
    set_global_seed(CONFIG["seed"])

    x_train, y_train, x_test, y_test = load_cifar10()
    x_train, y_train, x_val, y_val = split_train_val(x_train, y_train)

    base_datasets = make_tf_datasets(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        CONFIG["training"]["batch_size"],
        use_standardization=True,
        use_augmentation=True,
    )

    if args.mode in {"baseline", "all"}:
        run_baseline(base_datasets)
    if args.mode in {"stride_kernel", "all"}:
        run_stride_kernel_experiments(base_datasets)
    if args.mode in {"l2", "all"}:
        run_l2_experiments(base_datasets)
    if args.mode in {"preprocess", "all"}:
        run_preprocessing_ablation(x_train, y_train, x_val, y_val, x_test, y_test)


if __name__ == "__main__":
    main()

