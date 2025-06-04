# scripts/models/model_baseline_one.py

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils.datasets import create_dataloaders
from scripts.utils.logging import logging

from scripts.utils.eval_pred import evaluate_predictions  # local module (eval_pred.py)

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
parser.add_argument("--red_rat", default=16, type=int, help="Reduction ratio for the MLP hidden layer size.")
parser.add_argument("--drop1", default=0.4, type=float, help="Dropout rate in the 1st FC layer of the classification head.")
parser.add_argument("--drop2", default=0.4, type=float, help="Dropout rate in the 2nd FC layer of the classification head.")


##### ───────────────────────────────────────────────────────────── FULLY CONVOLUTIONAL NETWORK WITH CBAM ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
###   1. CONV → BN → GELU BLOCK
# ─────────────────────────────────────────────────────────────
class Conv_BN_GeLU(nn.Module):
    """
    2-D convolution + batch normalisation + GELU activation.
    """

    def __init__(
            self,
            in_chan: int,
            out_chan: int,
            k_size: Tuple[int, int],
            strd: Tuple[int, int] = (1, 1),
            pad: Tuple[int, int] = (0, 0),
            groups: int = 1,
            dilation: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan,
                              kernel_size=k_size,
                              stride=strd,
                              padding=pad,
                              dilation=dilation,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.act = nn.GELU()

    # ----------------- FORWARD PASS --------------------------
    def forward(self, x):
        logits = self.act(self.bn(self.conv(x)))
        return logits


# ─────────────────────────────────────────────────────────────
###   2.  CONVOLUTIONAL BLOCK ATTENTION MODULE
# ─────────────────────────────────────────────────────────────

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    (Woo et al., ECCV 2018)

    Args
    ----
    channels        : # feature maps of the incoming tensor
    reduction_ratio : channel squeeze factor  (default = 16)
    kernel_size     : spatial-attention conv kernel (default = 7)
    """

    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # ── Channel-attention sub-module ──────────────────────
        hidden = max(channels // reduction_ratio, 1)  # avoid 0
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        # ── Spatial-attention sub-module ─────────────────────
        self.conv_spat = nn.Conv2d(2, 1,  # 2 maps → 1
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2,
                                   bias=False)

        self.sigmoid = nn.Sigmoid()

    # ---------------------------------------------------------
    #  utilities
    # ---------------------------------------------------------
    @staticmethod  # decorator that tells Py not to pass instance or class as the 1st arg to the method
    def _channel_pool(x):
        """
        Channel attention summarizes feature-map activations across all spacial positions for each channel.
        """
        # global avg-pool  and  global max-pool over H×W
        avg = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B, C, 1, 1) -> (B,C)
        max = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B, C, 1, 1) -> (B,C)
        return avg, max

    @staticmethod
    def _spatial_pool(x):
        """
        Spacial attention summarizes feature-map activations across all channels for each spacial location.
        """
        # channel-wise avg/max  →  2×H×W descriptor
        avg = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        return torch.cat([avg, max], dim=1)  # (B,2,H,W)

    # ---------------------------------------------------------
    #  forward pass
    # ---------------------------------------------------------
    def forward(self, x):
        B, C, H, W = x.size()

        # Ordering from the paper
        # 1) CHANNEL attention --------------------------------
        avg, max = self._channel_pool(x)  # (B,C) each
        attn_ch = self.mlp(avg) + self.mlp(max)  # shared MLP
        attn_ch = self.sigmoid(attn_ch).view(B, C, 1, 1)  # broadcast
        x = x * attn_ch  # refine

        # 2) SPATIAL attention --------------------------------
        spat = self._spatial_pool(x)  # (B,2,H,W)
        attn_sp = self.sigmoid(self.conv_spat(spat))  # (B,1,H,W)
        x = x * attn_sp  # refine

        return x


# ─────────────────────────────────────────────────────────────
###   3. CNN WITH CBAM AND CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────

class FCNN(nn.Module):
    """
    Input:  (B, 1, 40, 218)      ── MFCC “image”
    Output: (B, 64, 9, 9)        ── compact feature map
    """

    def __init__(self, dropout1=0.2, dropout2=0.2, red_rat=16):
        super().__init__()

        # (a) Shrink x-axis from 218 → 109
        self.conv1 = Conv_BN_GeLU(1, 32,
                                  k_size=(3, 3),
                                  strd=(1, 2),
                                  pad=(1, 1))

        # (b) Two parallel 5×5 context CBs, stride (2,2)
        self.conv2_1 = Conv_BN_GeLU(32, 32,
                                    k_size=(5, 5),
                                    strd=(2, 2),
                                    pad=(0, 1))  # → 18 × 53

        self.conv_dil2_2 = Conv_BN_GeLU(32, 32,
                                        k_size=(5, 5),
                                        strd=(2, 2),
                                        pad=(2, 3),  # careful: keeps H,W
                                        dilation=2)  # 18 × 53

        # (c) Fuse & halve x-axis again: 53 → 27
        self.conv_fuse = Conv_BN_GeLU(64, 256,
                                      k_size=(3, 3),
                                      strd=(1, 2),
                                      pad=(1, 1))  # → 18 × 27

        # (d) 2×3 average pool to 9 × 9
        self.avgp = nn.AvgPool2d(kernel_size=(2, 3),
                                 stride=(2, 3))  # → 9 × 9

        # (e) CBAM attention (learned re-weighting)
        self.cbam = CBAM(channels=256, reduction_ratio=red_rat)

        # (f) Conv 3×3 valid → 7 × 7  (keeps channels)
        self.conv3 = Conv_BN_GeLU(256, 256,
                                  k_size=(3, 3),
                                  strd=(1, 1),
                                  pad=(0, 0))  # → 7 × 7

        # (g) MaxPool 2×2, stride 1 → 6 × 6
        self.maxp = nn.MaxPool2d(kernel_size=2,
                                 stride=1)  # → 6 × 6

        # (h) Conv 3×3 valid & channel squeeze: 256 → 128
        self.conv4 = Conv_BN_GeLU(256, 128,
                                  k_size=(3, 3),
                                  strd=(1, 1),
                                  pad=(0, 0))  # → 4 × 4

        # (i) Classifier head
        self.flatten = nn.Flatten()  # → 128·4·4 = 2048
        self.fc1 = nn.Linear(2048, 256, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=dropout1)

        self.fc2 = nn.Linear(256, 6)  # six emotion classes
        self.drop2 = nn.Dropout(p=dropout2)

    # ----------------- FORWARD PASS --------------------------
    def forward(self, x):
        x = self.conv1(x)  # (B,32,40,109)

        # context branches
        a = self.conv2_1(x)  # (B,32,18,53)
        b = self.conv_dil2_2(x)  # (B,32,18,53)
        x = torch.cat((a, b), dim=1)  # (B,64,18,53)

        x = self.conv_fuse(x)  # (B,256,18,27)
        x = self.avgp(x)  # (B,256, 9, 9)

        x = self.cbam(x)  # (B,256, 9, 9)

        x = self.conv3(x)  # (B,256, 7, 7)
        x = self.maxp(x)  # (B,256, 6, 6)

        x = self.conv4(x)  # (B,128, 4, 4)

        x = self.flatten(x)  # (B,2048)
        x = self.drop1(self.relu1(self.fc1(x)))  # (B,256)
        logits = self.drop2((self.fc2(x)))  # (B,  6)

        return logits


# ─────────────────────────────────────────────────────────────
###  Utility functions
# ─────────────────────────────────────────────────────────────


def set_torch_seed(seed: int, threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)


def xavier_init(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────
###  MAIN FUNCTION
# ─────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:  # noqa: C901  pylint: disable=too-many-locals,too-many-statements
    # 1) reproducibility & logging‑folder ------------------------------------------------
    set_torch_seed(args.seed, args.threads)

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)
    logging(args.logdir)

    # 2) data ---------------------------------------------------------------------------
    train_dl, dev_dl, test_dl = create_dataloaders(args.batch_size)

    # 3) model --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN(dropout1=args.drop1, dropout2=args.drop2, red_rat=args.red_rat).to(device)
    model.apply(xavier_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dl), eta_min=args.lr * 0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 4) training loop ------------------------------------------------------------------
    best_dev_acc = 0.0
    patience_counter, patience = 0, 5
    for epoch in range(args.epochs):
        # ‑‑ train phase ----------------------------------------------------------
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

        # ‑‑ dev phase ------------------------------------------------------------
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

        # ‑‑ checkpoint / early‑stopping -----------------------------------------
        if dev_acc > best_dev_acc + 1e-4:  # tiny delta avoids float noise
            best_dev_acc = dev_acc
            patience_counter = 0
            if epoch > 10:  # save only after some epochs
                torch.save(model.state_dict(), os.path.join(args.logdir, f"best_model_t{train_acc:.4f}_d{best_dev_acc:.4f}.pt"))
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

    # 5) test‑predictions ---------------------------------------------------------------
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

    # 6) automatic evaluation (requires ground‑truth labels) ----------------------------
    try:
        overall_acc, per_class_acc = evaluate_predictions(args.logdir)
        print(f"Evaluation – overall accuracy: {overall_acc:.4f}")
        print(f"Evaluation – per‑class accuracy: {per_class_acc}")
    except Exception as exc:  # broad, but we want to keep the run alive even if eval fails
        print(f"Post‑run evaluation skipped – {exc}")


if __name__ == "__main__":
    cli_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(cli_args)
