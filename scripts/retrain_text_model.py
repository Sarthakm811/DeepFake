from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.eval_text_dataset import SAMPLES
from src.text_detector import _resolve_text_model_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_emoji(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", text)
    return _normalize_spaces(text)


def build_augmented_labeled_samples(max_augs_per_sample: int = 3) -> tuple[list[str], list[int]]:
    clear_samples = [sample for sample in SAMPLES if sample["label"] is not None]

    texts: list[str] = []
    labels: list[int] = []

    for sample in clear_samples:
        text = sample["text"]
        label = int(sample["label"])

        variants = [
            text,
            _normalize_spaces(text),
            text.lower(),
            _strip_emoji(text),
            f"{text} Verified source reports this claim.",
            f"{text} Read before sharing.",
        ]

        seen = set()
        kept = 0
        for variant in variants:
            normalized = _normalize_spaces(variant)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            texts.append(normalized)
            labels.append(label)
            kept += 1
            if kept >= max_augs_per_sample:
                break

    return texts, labels


@dataclass
class EvalMetrics:
    accuracy: float
    real_precision: float
    fake_precision: float
    real_recall: float
    fake_recall: float
    recall_gap_abs: float


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(y_true: list[int], y_pred: list[int]) -> EvalMetrics:
    total = len(y_true)
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    accuracy = (tp + tn) / total if total else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0

    return EvalMetrics(
        accuracy=accuracy,
        real_precision=real_precision,
        fake_precision=fake_precision,
        real_recall=real_recall,
        fake_recall=fake_recall,
        recall_gap_abs=abs(real_recall - fake_recall),
    )


def evaluate_model(
    model: DistilBertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return compute_metrics(y_true, y_pred)


def save_model_artifacts(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    target_dir: Path,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for stale_name in ["model.safetensors", "model-00001-of-00001.safetensors", "model.safetensors.index.json"]:
        stale_path = target_dir / stale_name
        if stale_path.exists():
            try:
                stale_path.unlink()
            except OSError:
                pass

    model.config.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    weights_path = target_dir / "pytorch_model.bin"
    tmp_weights_path = target_dir / "pytorch_model.bin.tmp"
    torch.save(model.state_dict(), tmp_weights_path)
    shutil.move(str(tmp_weights_path), str(weights_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain DistilBERT text detector with fairness-aware validation")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-augs", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)

    base_model_path = _resolve_text_model_path(None)
    output_dir = PROJECT_ROOT / "models" / "text_distilbert"
    notebook_output_dir = PROJECT_ROOT / "notebooks" / "models" / "text_distilbert"
    report_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = build_augmented_labeled_samples(max_augs_per_sample=args.max_augs)
    if len(set(labels)) < 2:
        raise RuntimeError("Training data must contain both classes.")

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(base_model_path)

    train_enc = tokenizer(
        x_train,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        x_val,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    train_ds = EncodedTextDataset(train_enc, y_train)
    val_ds = EncodedTextDataset(val_enc, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_objective = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)

        balanced_acc = 0.5 * (val_metrics.real_recall + val_metrics.fake_recall)
        objective = balanced_acc - 0.2 * val_metrics.recall_gap_abs

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_accuracy": float(val_metrics.accuracy),
                "val_real_precision": float(val_metrics.real_precision),
                "val_fake_precision": float(val_metrics.fake_precision),
                "val_real_recall": float(val_metrics.real_recall),
                "val_fake_recall": float(val_metrics.fake_recall),
                "val_recall_gap_abs": float(val_metrics.recall_gap_abs),
                "val_objective": float(objective),
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
            f"acc={val_metrics.accuracy:.4f} | gap={val_metrics.recall_gap_abs:.4f}"
        )

        if objective > best_objective:
            best_objective = objective
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    save_model_artifacts(model, tokenizer, output_dir)
    save_model_artifacts(model, tokenizer, notebook_output_dir)

    final_val_metrics = evaluate_model(model, val_loader, device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"text_retrain_report_{ts}.json"

    report = {
        "base_model_path": str(base_model_path),
        "saved_model_path": str(output_dir),
        "saved_notebook_model_path": str(notebook_output_dir),
        "device": str(device),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_augs_per_sample": args.max_augs,
        "best_objective": float(best_objective),
        "final_val_metrics": {
            "accuracy": float(final_val_metrics.accuracy),
            "real_precision": float(final_val_metrics.real_precision),
            "fake_precision": float(final_val_metrics.fake_precision),
            "real_recall": float(final_val_metrics.real_recall),
            "fake_recall": float(final_val_metrics.fake_recall),
            "recall_gap_abs": float(final_val_metrics.recall_gap_abs),
        },
        "history": history,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== TEXT RETRAIN COMPLETE ===")
    print(f"Train samples: {len(x_train)} | Val samples: {len(x_val)}")
    print(f"Saved model: {output_dir}")
    print(f"Saved notebook model: {notebook_output_dir}")
    print(f"Val accuracy: {final_val_metrics.accuracy:.4f}")
    print(f"Real recall: {final_val_metrics.real_recall:.4f}")
    print(f"Fake recall: {final_val_metrics.fake_recall:.4f}")
    print(f"Recall gap: {final_val_metrics.recall_gap_abs:.4f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
