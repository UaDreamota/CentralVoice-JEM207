# scripts/models/no_cbam/baseline.py

# ─────────────────────────────────────────────────────────────
### IMPORTS
# ─────────────────────────────────────────────────────────────

import argparse
import datetime
import os
import re
import csv
import random
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.utils.datasets import create_dataloaders
from scripts.utils.logging import logging
from scripts.utils.eval_pred import evaluate_predictions  

# ─────────────────────────────────────────────────────────────
### DYNAMIC ARGUMENTS
# ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=24, type=int, help="Batch size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--label_smoothing", default=0.05, type=float, help="Label smoothing.")

##### ───────────────────────────────────────────────────────────── BASELINE CNN ─────────────────────────────────────────────────────────────


class SimpleCNN(nn.Module):
    """Minimal CNN: two conv blocks followed by global average pooling."""

    def __init__(
            self,
            in_chan: int,
            out_chan: int,
            k_size: Tuple[int, int],
            pad: Tuple[int, int],
            drop: float
            ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_chan * 2, out_chan * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chan * 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_chan * 4, out_chan * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_chan * 2)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_chan * 4, out_chan * 4)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop)

        self.class_l = nn.Linear(out_chan * 4, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.drop(self.relu_fc(self.fc(x)))
        logits = self.class_l(x)
        
        return logits


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_torch_seed(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)


def xavier_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Main training / evaluation routine
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:  # noqa: C901
    # 1) reproducibility & logging‑folder -------------------------------------
    set_torch_seed(args.seed, args.threads)

    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(
                        re.sub("(.)[^_]*_?", r"\1", k), v
                    )
                    for k, v in sorted(vars(args).items())
                )
            ),
        ),
    )
    os.makedirs(args.logdir, exist_ok=True)
    logging(args.logdir)

    # 2) data ---------------------------------------------------------------
    train_dl, dev_dl, test_dl = create_dataloaders(args.batch_size)

    # 3) model --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.apply(xavier_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_dl), eta_min=args.lr * 0.01
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 4) training loop ------------------------------------------------------
    best_dev_acc = 0.0
    patience_counter, patience = 0, 5
    for epoch in range(args.epochs):
        # Train phase -------------------------------------------------------
        model.train()
        epoch_loss, batches = 0.0, 0
        train_correct = train_total = 0
        for feats, label in train_dl:
            feats, label = feats.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(feats)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            batches += 1
            train_correct += (out.argmax(dim=1) == label).sum().item()
            train_total += label.size(0)

        train_acc = train_correct / train_total if train_total else 0.0
        train_loss_mean = epoch_loss / batches

        # Dev phase ---------------------------------------------------------
        model.eval()
        correct = total = val_batches = 0
        val_loss = 0.0
        with torch.no_grad():
            for feats, label in dev_dl:
                feats, label = feats.to(device), label.to(device)
                outputs = model(feats)
                val_loss += loss_fn(outputs, label).item()
                correct += (outputs.argmax(dim=1) == label).sum().item()
                total += label.size(0)
                val_batches += 1
        dev_acc = correct / total if total else 0.0

        # Checkpoint / early stopping --------------------------------------
        if dev_acc > best_dev_acc + 1e-4:
            best_dev_acc = dev_acc
            patience_counter = 0
            if epoch > 10:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.logdir,
                        f"best_model_t{train_acc:.4f}_d{best_dev_acc:.4f}.pt",
                    ),
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"No dev‑accuracy gain for {patience} epochs – early stopping at epoch {epoch + 1}. "
                    f"Best dev accuracy: {best_dev_acc:.4f}"
                )
                break

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}: train_loss {train_loss_mean:.4f}, train_acc {train_acc:.4f}, "
            f"dev_acc {dev_acc:.4f}, lr {current_lr:.6f}"
        )

    # 5) test predictions ---------------------------------------------------
    ckpts = sorted(Path(args.logdir).glob("best_model_*.pt"))
    if ckpts:
        def _dev_acc(path: Path) -> float:
            m = re.search(r"_d([0-9]*\.[0-9]+)\.pt$", path.name)
            return float(m.group(1)) if m else -1.0

        # Prefer highest dev acc; break ties by most recent mtime
        best_ckpt = max(ckpts, key=lambda p: (_dev_acc(p), p.stat().st_mtime))
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best checkpoint: {best_ckpt}")
    else:
        print("No best_model_*.pt found – using last in-memory weights.")

    model.eval()
    test_preds = []
    with torch.no_grad():
        for feats, _ in test_dl:
            preds = model(feats.to(device)).argmax(dim=1).cpu().tolist()
            test_preds.extend(preds)

    pred_file = os.path.join(args.logdir, "predictions.csv")
    with open(pred_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction"])
        for i, p in enumerate(test_preds):
            writer.writerow([f"sample_{i}", p])
    print(f"Predictions saved to {pred_file}")


    # 6) automatic evaluation ----------------------------------------------
    try:
        overall_acc, per_class_acc = evaluate_predictions(args.logdir)
        print(f"Evaluation – overall accuracy: {overall_acc:.4f}")
        print(f"Evaluation – per‑class accuracy: {per_class_acc}")
    except Exception as exc:  # keep the run alive even if eval fails
        print(f"Post‑run evaluation skipped – {exc}")


if __name__ == "__main__":
    cli_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(cli_args)