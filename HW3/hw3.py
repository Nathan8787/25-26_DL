import os
import json
import csv
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch import amp


# ============================================================
# Helpers: seed, dirs, accuracy
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct, total


# ============================================================
# Configuration & Experiments
# ============================================================


@dataclass
class Config:
    train_file: str = "shakespeare_train.txt"
    valid_file: str = "shakespeare_valid.txt"
    batch_size: int = 512
    seq_len: int = 100
    model_type: str = "RNN"  # "RNN" or "LSTM"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.25
    learning_rate: float = 0.002
    epochs: int = 5
    patience: int = 5  # will be max(patience, epochs) to guarantee breakpoints
    clip_grad_norm: float = 5.0
    seed: int = 113024510
    plots_dir: str = "plots"
    outputs_dir: str = "outputs"
    checkpoints_dir: str = "checkpoints"
    ckpt_path: str = ""
    metrics_path: str = ""
    breakpoints_path: str = ""
    arch_path: str = ""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def summary(self) -> str:
        items = {
            "device": str(self.device),
            "seed": self.seed,
            "train_file": self.train_file,
            "valid_file": self.valid_file,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "patience": self.patience,
            "clip_grad_norm": self.clip_grad_norm,
            "ckpt_path": self.ckpt_path,
            "metrics_path": self.metrics_path,
            "breakpoints_path": self.breakpoints_path,
            "arch_path": self.arch_path,
        }
        return " | ".join([f"{k}={v}" for k, v in items.items()])


def build_experiments(base_cfg: Config) -> List[Dict]:
    exps: List[Dict] = []
    # Baseline RNN
    exps.append(
        {
            "name": "RNN",
            "model_type": "RNN",
            "hidden_size": 128,
            "seq_len": 100,
        }
    )
    # RNN hidden size comparison
    for hs in [64, 256]:
        exps.append(
            {
                "name": f"RNN_HS_{hs}",
                "model_type": "RNN",
                "hidden_size": hs,
                "seq_len": 100,
            }
        )
    # RNN sequence length comparison
    for sl in [50, 150]:
        exps.append(
            {
                "name": f"RNN_SL_{sl}",
                "model_type": "RNN",
                "hidden_size": 128,
                "seq_len": sl,
            }
        )
    # Baseline LSTM
    exps.append(
        {
            "name": "LSTM",
            "model_type": "LSTM",
            "hidden_size": 128,
            "seq_len": 100,
        }
    )
    # LSTM hidden size comparison
    for hs in [64, 256]:
        exps.append(
            {
                "name": f"LSTM_HS_{hs}",
                "model_type": "LSTM",
                "hidden_size": hs,
                "seq_len": 100,
            }
        )
    # LSTM sequence length comparison
    for sl in [50, 150]:
        exps.append(
            {
                "name": f"LSTM_SL_{sl}",
                "model_type": "LSTM",
                "hidden_size": 128,
                "seq_len": sl,
            }
        )

    # Attach paths
    for exp in exps:
        name = exp["name"]
        exp["ckpt_path"] = os.path.join(base_cfg.checkpoints_dir, f"best_model_{name}.pth")
        exp["metrics_path"] = os.path.join(base_cfg.outputs_dir, f"metrics_{name}")
        exp["breakpoints_path"] = os.path.join(base_cfg.outputs_dir, f"breakpoints_{name}.txt")
        exp["arch_path"] = os.path.join(base_cfg.outputs_dir, f"arch_{name}.txt")
    return exps


# ============================================================
# Data
# ============================================================


class ShakespeareDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int, vocab=None, char_to_idx=None, idx_to_char=None):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        if vocab is None:
            self.vocab = sorted(list(set(self.text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        else:
            self.vocab = vocab
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
        self.vocab_size = len(self.vocab)
        self.data = [self.char_to_idx[ch] for ch in self.text]

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)
        return input_seq, target_seq


# ============================================================
# Model
# ============================================================


class CharRNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.model_type = config.model_type
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(config.input_size, config.hidden_size)
        if self.model_type == "RNN":
            self.rnn = nn.RNN(
                config.hidden_size,
                config.hidden_size,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout,
            )
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                config.hidden_size,
                config.hidden_size,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout,
            )
        else:
            raise ValueError("Invalid model_type. Choose 'RNN' or 'LSTM'.")
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: torch.Tensor, hidden):
        embeds = self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        weight = next(self.parameters()).data
        if self.model_type == "LSTM":
            return (
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            )
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device)


# ============================================================
# Training / Evaluation
# ============================================================


def save_metrics(metrics: List[Dict], base_path: str) -> None:
    csv_path = base_path + ".csv"
    json_path = base_path + ".json"
    if metrics:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved: {csv_path}, {json_path}")


def plot_curves(losses: Dict[str, Dict[str, List[float]]], ylabel: str, title: str, save_path: str) -> None:
    plt.figure(figsize=(10, 5))
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i, (name, series) in enumerate(losses.items()):
        color = colors[i % len(colors)]
        if "train" in series:
            plt.plot(series["train"], "--", label=f"{name} Train", color=color, alpha=0.7)
        if "valid" in series:
            plt.plot(series["valid"], "-", label=f"{name} Valid", color=color, linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved: {save_path}")
    plt.close()


def create_breakpoints(epochs: int) -> List[int]:
    arr = np.linspace(1, epochs, 5).round().astype(int).tolist()
    uniq = sorted(list(dict.fromkeys(arr)))
    while len(uniq) < 5:
        uniq.append(epochs)
    return uniq[:5]


def generate_text(
    model: CharRNN,
    start_str: str,
    length: int,
    temperature: float,
    config: Config,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
) -> str:
    model.eval()
    input_seq = [char_to_idx.get(ch, 0) for ch in start_str]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(config.device)
    generated = start_str
    hidden = model.init_hidden(1, config.device)
    use_amp = config.device.type == "cuda"
    with torch.no_grad():
        for _ in range(length):
            with amp.autocast(device_type="cuda", enabled=use_amp):
                output, hidden = model(input_tensor, hidden)
            logits = output[-1]
            if temperature == 0:
                predicted_idx = torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits / temperature, dim=0)
                predicted_idx = torch.multinomial(probs, 1).item()
            predicted_char = idx_to_char[predicted_idx]
            generated += predicted_char
            input_tensor = torch.tensor([[predicted_idx]], dtype=torch.long).to(config.device)
    return generated


def format_to_line_count(text: str, desired_lines: int = 12) -> Tuple[str, int]:
    desired_lines = max(10, min(15, desired_lines))
    if desired_lines <= 1:
        return text, 1
    chunk = max(1, math.ceil(len(text) / desired_lines))
    lines = [text[i : i + chunk] for i in range(0, len(text), chunk)]
    lines = lines[:desired_lines] if len(lines) > desired_lines else lines
    formatted = "\n".join(lines)
    return formatted, len(lines)


def train_model(
    model: CharRNN,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Config,
    model_name: str,
    seed_text: str = "The ",
    gen_length: int = 100,
    temperature: float = 0.6,
) -> Tuple[List[Dict], List[str]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler(enabled=config.device.type == "cuda")
    patience = max(config.patience, config.epochs)
    breakpoints = create_breakpoints(config.epochs)
    metrics: List[Dict] = []
    breakpoint_outputs: List[str] = []
    best_valid = float("inf")
    best_state = None

    print(f"[START] {model_name} | {config.summary()}")
    print(model)
    with open(config.arch_path, "w", encoding="utf-8") as f:
        f.write(str(model))

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        for inputs, targets in pbar:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size, config.device)
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()
            optimizer.zero_grad()
            with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                output, hidden = model(inputs, hidden)
                loss = criterion(output, targets.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            c, t = compute_accuracy(output, targets.view(-1))
            correct_train += c
            total_train += t
            bpc = loss.item() / np.log(2)
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1), "bpc": bpc})
        avg_train_loss = running_loss / len(train_loader)
        train_bpc = avg_train_loss / np.log(2)
        train_acc = correct_train / max(1, total_train)
        train_err = 1.0 - train_acc

        # Validation
        model.eval()
        val_loss_sum = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                batch_size = inputs.size(0)
                hidden = model.init_hidden(batch_size, config.device)
                with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                    output, hidden = model(inputs, hidden)
                    loss = criterion(output, targets.view(-1))
                val_loss_sum += loss.item()
                c, t = compute_accuracy(output, targets.view(-1))
                correct_val += c
                total_val += t
        avg_val_loss = val_loss_sum / len(valid_loader)
        val_bpc = avg_val_loss / np.log(2)
        val_acc = correct_val / max(1, total_val)
        val_err = 1.0 - val_acc

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_bpc": train_bpc,
                "train_acc": train_acc,
                "train_err": train_err,
                "valid_loss": avg_val_loss,
                "valid_bpc": val_bpc,
                "valid_acc": val_acc,
                "valid_err": val_err,
            }
        )
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss {avg_train_loss:.4f} | Train BPC {train_bpc:.4f} | Train Acc {train_acc:.4f} | Train Err {train_err:.4f} | "
            f"Valid Loss {avg_val_loss:.4f} | Valid BPC {val_bpc:.4f} | Valid Acc {val_acc:.4f} | Valid Err {val_err:.4f}"
        )

        # Breakpoints
        if epoch in breakpoints:
            ds = train_loader.dataset
            gen = generate_text(
                model,
                start_str=seed_text,
                length=gen_length,
                temperature=temperature,
                config=config,
                char_to_idx=ds.char_to_idx,
                idx_to_char=ds.idx_to_char,
            )
            block = (
                f"[Breakpoint] model={model_name} | epoch={epoch} | model_type={config.model_type} | "
                f"hidden_size={config.hidden_size} | seq_len={config.seq_len} | num_layers={config.num_layers} | "
                f"dropout={config.dropout} | temperature={temperature} | seed_text='{seed_text}'\n{gen}\n\n"
            )
            breakpoint_outputs.append(block)
            print(block)

        # Track best
        if avg_val_loss < best_valid:
            best_valid = avg_val_loss
            best_state = model.state_dict()
            torch.save(best_state, config.ckpt_path)

        # No early stop before epochs because patience >= epochs
        if epoch >= patience:
            pass

    # Save breakpoint outputs
    with open(config.breakpoints_path, "w", encoding="utf-8") as f:
        for block in breakpoint_outputs:
            f.write(block)
    print(f"Breakpoints saved: {config.breakpoints_path}")

    # Save metrics
    save_metrics(metrics, config.metrics_path)
    return metrics, breakpoint_outputs


# ============================================================
# Summary helper
# ============================================================


def write_summary_rnn_vs_lstm(
    outputs_dir: str,
    metrics_map: Dict[str, List[Dict]],
    plot_refs: List[str],
) -> None:
    path = os.path.join(outputs_dir, "summary_rnn_vs_lstm.txt")
    def best_row(name: str) -> Dict:
        rows = metrics_map.get(name, [])
        if not rows:
            return {}
        return min(rows, key=lambda r: r["valid_loss"])
    best_rnn = best_row("RNN")
    best_lstm = best_row("LSTM")
    with open(path, "w", encoding="utf-8") as f:
        f.write("RNN vs LSTM Summary (numeric)\n")
        f.write(f"Best RNN: {best_rnn}\n")
        f.write(f"Best LSTM: {best_lstm}\n")
        f.write("\nPlot references:\n")
        for p in plot_refs:
            f.write(f"- {p}\n")
        f.write("\n[Post 4-1 Placeholder] Architecture/metric comparison remarks:\n")
        f.write("...\n")
        f.write("\n[Post 4-2 Placeholder] Breakpoint-generated samples comparison:\n")
        f.write("...\n")
        f.write("\n[Post 4-3 Placeholder] Hyperparameter effects (hidden size/seq_len) comparison:\n")
        f.write("...\n")
    print(f"Summary saved: {path}")


# ============================================================
# Main
# ============================================================


def main():
    cfg = Config()
    ensure_dir(cfg.plots_dir)
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.checkpoints_dir)
    set_seed(cfg.seed)
    print(f"Using device: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(cfg.train_file) or not os.path.exists(cfg.valid_file):
        print("Error: Data files not found.")
        return

    base_train_ds = ShakespeareDataset(cfg.train_file, cfg.seq_len)
    base_valid_ds = ShakespeareDataset(
        cfg.valid_file,
        cfg.seq_len,
        vocab=base_train_ds.vocab,
        char_to_idx=base_train_ds.char_to_idx,
        idx_to_char=base_train_ds.idx_to_char,
    )

    experiments = build_experiments(cfg)
    metrics_store: Dict[str, List[Dict]] = {}

    # Containers for comparisons
    loss_plots_hidden_rnn = {}
    loss_plots_seq_rnn = {}
    loss_plots_hidden_lstm = {}
    loss_plots_seq_lstm = {}
    loss_plots_rnn_vs_lstm = {}
    err_plots_hidden_rnn = {}
    err_plots_seq_rnn = {}
    err_plots_hidden_lstm = {}
    err_plots_seq_lstm = {}
    err_plots_rnn_vs_lstm = {}
    bpc_plots_hidden_rnn = {}
    bpc_plots_seq_rnn = {}
    bpc_plots_hidden_lstm = {}
    bpc_plots_seq_lstm = {}
    bpc_plots_rnn_vs_lstm = {}

    for exp in experiments:
        # Clone config per experiment
        exp_cfg = Config(**asdict(cfg))
        exp_cfg.model_type = exp["model_type"]
        exp_cfg.hidden_size = exp["hidden_size"]
        exp_cfg.seq_len = exp["seq_len"]
        exp_cfg.ckpt_path = exp["ckpt_path"]
        exp_cfg.metrics_path = exp["metrics_path"]
        exp_cfg.breakpoints_path = exp["breakpoints_path"]
        exp_cfg.arch_path = exp["arch_path"]
        exp_cfg.patience = max(exp_cfg.patience, exp_cfg.epochs)

        # Dataset (reuse vocab, adjust seq_len if needed)
        if exp_cfg.seq_len == cfg.seq_len:
            train_ds = base_train_ds
            valid_ds = base_valid_ds
        else:
            train_ds = ShakespeareDataset(cfg.train_file, exp_cfg.seq_len, vocab=base_train_ds.vocab, char_to_idx=base_train_ds.char_to_idx, idx_to_char=base_train_ds.idx_to_char)
            valid_ds = ShakespeareDataset(cfg.valid_file, exp_cfg.seq_len, vocab=base_train_ds.vocab, char_to_idx=base_train_ds.char_to_idx, idx_to_char=base_train_ds.idx_to_char)

        exp_cfg.input_size = train_ds.vocab_size  # type: ignore
        exp_cfg.output_size = train_ds.vocab_size  # type: ignore

        train_loader = DataLoader(train_ds, batch_size=exp_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=exp_cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

        model = CharRNN(exp_cfg).to(exp_cfg.device)

        metrics, _ = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=exp_cfg,
            model_name=exp["name"],
        )
        metrics_store[exp["name"]] = metrics

        # Build plot feeds
        train_loss_series = [m["train_loss"] for m in metrics]
        valid_loss_series = [m["valid_loss"] for m in metrics]
        train_err_series = [m["train_err"] for m in metrics]
        valid_err_series = [m["valid_err"] for m in metrics]
        train_bpc_series = [m["train_bpc"] for m in metrics]
        valid_bpc_series = [m["valid_bpc"] for m in metrics]

        pair_loss = {"train": train_loss_series, "valid": valid_loss_series}
        pair_err = {"train": train_err_series, "valid": valid_err_series}
        pair_bpc = {"train": train_bpc_series, "valid": valid_bpc_series}

        if exp_cfg.model_type == "RNN":
            loss_plots_rnn_vs_lstm["RNN"] = pair_loss
            err_plots_rnn_vs_lstm["RNN"] = pair_err
            bpc_plots_rnn_vs_lstm["RNN"] = pair_bpc
            if exp["name"].startswith("RNN_HS"):
                loss_plots_hidden_rnn[exp["name"]] = pair_loss
                err_plots_hidden_rnn[exp["name"]] = pair_err
                bpc_plots_hidden_rnn[exp["name"]] = pair_bpc
            elif exp["name"].startswith("RNN_SL"):
                loss_plots_seq_rnn[exp["name"]] = pair_loss
                err_plots_seq_rnn[exp["name"]] = pair_err
                bpc_plots_seq_rnn[exp["name"]] = pair_bpc
            else:
                loss_plots_hidden_rnn["RNN_128"] = pair_loss
                loss_plots_seq_rnn["RNN_Seq100"] = pair_loss
                err_plots_hidden_rnn["RNN_128"] = pair_err
                err_plots_seq_rnn["RNN_Seq100"] = pair_err
                bpc_plots_hidden_rnn["RNN_128"] = pair_bpc
                bpc_plots_seq_rnn["RNN_Seq100"] = pair_bpc
        else:
            loss_plots_rnn_vs_lstm["LSTM"] = pair_loss
            err_plots_rnn_vs_lstm["LSTM"] = pair_err
            bpc_plots_rnn_vs_lstm["LSTM"] = pair_bpc
            if exp["name"].startswith("LSTM_HS"):
                loss_plots_hidden_lstm[exp["name"]] = pair_loss
                err_plots_hidden_lstm[exp["name"]] = pair_err
                bpc_plots_hidden_lstm[exp["name"]] = pair_bpc
            elif exp["name"].startswith("LSTM_SL"):
                loss_plots_seq_lstm[exp["name"]] = pair_loss
                err_plots_seq_lstm[exp["name"]] = pair_err
                bpc_plots_seq_lstm[exp["name"]] = pair_bpc
            else:
                loss_plots_hidden_lstm["LSTM_128"] = pair_loss
                loss_plots_seq_lstm["LSTM_Seq100"] = pair_loss
                err_plots_hidden_lstm["LSTM_128"] = pair_err
                err_plots_seq_lstm["LSTM_Seq100"] = pair_err
                bpc_plots_hidden_lstm["LSTM_128"] = pair_bpc
                bpc_plots_seq_lstm["LSTM_Seq100"] = pair_bpc

        print(f"[END] {exp['name']} artifacts:")
        print(f"- checkpoint: {exp_cfg.ckpt_path}")
        print(f"- metrics: {exp_cfg.metrics_path}.csv / .json")
        print(f"- breakpoints: {exp_cfg.breakpoints_path}")
        print(f"- arch: {exp_cfg.arch_path}")

    # Plots
    plot_curves(loss_plots_hidden_rnn, "Loss", "RNN Hidden Size Comparison (Loss)", os.path.join(cfg.plots_dir, "comparison_hidden_size.png"))
    plot_curves(err_plots_hidden_rnn, "Error Rate", "RNN Hidden Size Comparison (Error)", os.path.join(cfg.plots_dir, "comparison_hidden_size_error.png"))
    plot_curves(bpc_plots_hidden_rnn, "BPC", "RNN Hidden Size Comparison (BPC)", os.path.join(cfg.plots_dir, "comparison_hidden_size_bpc.png"))

    plot_curves(loss_plots_seq_rnn, "Loss", "RNN Sequence Length Comparison (Loss)", os.path.join(cfg.plots_dir, "comparison_seq_len.png"))
    plot_curves(err_plots_seq_rnn, "Error Rate", "RNN Sequence Length Comparison (Error)", os.path.join(cfg.plots_dir, "comparison_seq_len_error.png"))
    plot_curves(bpc_plots_seq_rnn, "BPC", "RNN Sequence Length Comparison (BPC)", os.path.join(cfg.plots_dir, "comparison_seq_len_bpc.png"))

    plot_curves(loss_plots_hidden_lstm, "Loss", "LSTM Hidden Size Comparison (Loss)", os.path.join(cfg.plots_dir, "comparison_lstm_hidden_size.png"))
    plot_curves(err_plots_hidden_lstm, "Error Rate", "LSTM Hidden Size Comparison (Error)", os.path.join(cfg.plots_dir, "comparison_lstm_hidden_size_error.png"))
    plot_curves(bpc_plots_hidden_lstm, "BPC", "LSTM Hidden Size Comparison (BPC)", os.path.join(cfg.plots_dir, "comparison_lstm_hidden_size_bpc.png"))

    plot_curves(loss_plots_seq_lstm, "Loss", "LSTM Sequence Length Comparison (Loss)", os.path.join(cfg.plots_dir, "comparison_lstm_seq_len.png"))
    plot_curves(err_plots_seq_lstm, "Error Rate", "LSTM Sequence Length Comparison (Error)", os.path.join(cfg.plots_dir, "comparison_lstm_seq_len_error.png"))
    plot_curves(bpc_plots_seq_lstm, "BPC", "LSTM Sequence Length Comparison (BPC)", os.path.join(cfg.plots_dir, "comparison_lstm_seq_len_bpc.png"))

    plot_curves(loss_plots_rnn_vs_lstm, "Loss", "RNN vs LSTM (Loss)", os.path.join(cfg.plots_dir, "comparison_rnn_vs_lstm.png"))
    plot_curves(err_plots_rnn_vs_lstm, "Error Rate", "RNN vs LSTM (Error)", os.path.join(cfg.plots_dir, "comparison_rnn_vs_lstm_error.png"))
    plot_curves(bpc_plots_rnn_vs_lstm, "BPC", "RNN vs LSTM (BPC)", os.path.join(cfg.plots_dir, "comparison_rnn_vs_lstm_bpc.png"))

    # Prime generation using best LSTM
    lstm_ckpt = os.path.join(cfg.checkpoints_dir, "best_model_LSTM.pth")
    if os.path.exists(lstm_ckpt):
        prime_cfg = Config(**asdict(cfg))
        prime_cfg.model_type = "LSTM"
        prime_cfg.hidden_size = 128
        prime_cfg.seq_len = 100
        prime_cfg.input_size = base_train_ds.vocab_size  # type: ignore
        prime_cfg.output_size = base_train_ds.vocab_size  # type: ignore
        prime_model = CharRNN(prime_cfg).to(prime_cfg.device)
        prime_model.load_state_dict(torch.load(lstm_ckpt, map_location=prime_cfg.device))
        prime_word = "JULIET"
        raw_gen = generate_text(
            prime_model,
            start_str=prime_word,
            length=500,
            temperature=0.6,
            config=prime_cfg,
            char_to_idx=base_train_ds.char_to_idx,
            idx_to_char=base_train_ds.idx_to_char,
        )
        formatted_gen, line_count = format_to_line_count(raw_gen, desired_lines=12)
        gen_path = os.path.join(cfg.outputs_dir, "generation_final.txt")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(f"Prime: {prime_word}\n")
            f.write(f"Lines: {line_count}\n\n")
            f.write(formatted_gen)
        print(f"Prime generation saved: {gen_path}")
    else:
        print("LSTM checkpoint for prime generation not found; skip prime generation.")

    # Summary
    summary_plots = [
        "comparison_rnn_vs_lstm.png",
        "comparison_rnn_vs_lstm_error.png",
        "comparison_rnn_vs_lstm_bpc.png",
    ]
    write_summary_rnn_vs_lstm(cfg.outputs_dir, metrics_store, summary_plots)
    print("All experiments complete.")


if __name__ == "__main__":
    main()
