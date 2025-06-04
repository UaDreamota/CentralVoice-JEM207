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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from scripts.utils.datasets import create_dataloaders

# ─────────────────────────────────────────────────────────────
### DYNAMIC ARGUMENTS
# ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.02, type = float, help = "Learning rate.")
parser.add_argument("--label_smoothing", default=0.05, type = float, help = "Label smoothing.")
parser.add_argument("--red_rat", default=16, type = int, help="Reduction ratio for the MLP hidden layer size.")
parser.add_argument("--drop1", default=0.2, type=float, help="Dropout rate in the 1st FC layer of the classification head.")
parser.add_argument("--drop2", default=0.2, type=float, help="Dropout rate in the 2nd FC layer of the classification head.")

    
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
            in_chan:    int,
            out_chan:   int,
            k_size:     Tuple[int,int],
            strd:       Tuple[int,int]=(1,1),
            pad:        Tuple[int,int]=(0,0),
            groups:     int=1,
            dilation:   int=1
            ):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan,
                              kernel_size=k_size,
                              stride=strd,
                              padding=pad,
                              dilation=dilation,
                              groups=groups,
                              bias=False)          
        self.bn   = nn.BatchNorm2d(out_chan)
        self.act  = nn.GELU()

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
        hidden = max(channels // reduction_ratio, 1)          # avoid 0
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        # ── Spatial-attention sub-module ─────────────────────
        self.conv_spat = nn.Conv2d(2, 1,                  # 2 maps → 1
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2,
                                   bias=False)

        self.sigmoid = nn.Sigmoid()

    # ---------------------------------------------------------
    #  utilities
    # ---------------------------------------------------------
    @staticmethod # decorator that tells Py not to pass instance or class as the 1st arg to the method
    def _channel_pool(x):
        """
        Channel attention summarizes feature-map activations across all spacial positions for each channel.
        """
        # global avg-pool  and  global max-pool over H×W
        avg = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B, C, 1, 1) -> (B,C)
        max = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1) # (B, C, 1, 1) -> (B,C)
        return avg, max

    @staticmethod
    def _spatial_pool(x):
        """
        Spacial attention summarizes feature-map activations across all channels for each spacial location.
        """
        # channel-wise avg/max  →  2×H×W descriptor
        avg = torch.mean(x, dim=1, keepdim=True)                 # (B,1,H,W)
        max, _ = torch.max(x, dim=1, keepdim=True)               # (B,1,H,W)
        return torch.cat([avg, max], dim=1)                      # (B,2,H,W)

    # ---------------------------------------------------------
    #  forward pass
    # ---------------------------------------------------------
    def forward(self, x):
        B, C, H, W = x.size()

        # Ordering from the paper
        # 1) CHANNEL attention --------------------------------
        avg, max = self._channel_pool(x)                      # (B,C) each
        attn_ch = self.mlp(avg) + self.mlp(max)               # shared MLP
        attn_ch = self.sigmoid(attn_ch).view(B, C, 1, 1)      # broadcast
        x = x * attn_ch                                       # refine

        # 2) SPATIAL attention --------------------------------
        spat = self._spatial_pool(x)                          # (B,2,H,W)
        attn_sp = self.sigmoid(self.conv_spat(spat))          # (B,1,H,W)
        x = x * attn_sp                                       # refine

        return x


# ─────────────────────────────────────────────────────────────
###   3. CNN WITH CBAM AND CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────

class FCNN(nn.Module):
    """
    Input:  (B, 1, 40, 218)      ── MFCC “image”
    Output: (B, 64, 9, 9)        ── compact feature map
    """
    def __init__(self, dropout1 = 0.2, dropout2 = 0.2, red_rat = 16):
        super().__init__()

        # (a) Shrink x-axis from 218 → 109
        self.conv1 = Conv_BN_GeLU(1, 32,
                                      k_size=(3,3),
                                      strd=(1,2),
                                      pad=(1,1))

        # (b) Two parallel 5×5 context CBs, stride (2,2)
        self.conv2_1 = Conv_BN_GeLU(32, 32,
                                  k_size=(5,5),
                                  strd=(2,2),
                                  pad=(0,1))           # → 18 × 53

        self.conv_dil2_2 = Conv_BN_GeLU(32, 32,
                                  k_size=(5,5),
                                  strd=(2,2),
                                  pad=(2,3),            # careful: keeps H,W
                                  dilation=2)           #     18 × 53

        # (c) Fuse & halve x-axis again: 53 → 27
        self.conv_fuse = Conv_BN_GeLU(64, 256,
                                    k_size=(3,3),
                                    strd=(1,2),
                                    pad=(1,1))          # → 18 × 27

        # (d) 2×3 average pool to 9 × 9
        self.avgp = nn.AvgPool2d(kernel_size=(2,3),
                                stride=(2,3))            # → 9 × 9
        
        # (e) CBAM attention (learned re-weighting)
        self.cbam = CBAM(channels=256, reduction_ratio=red_rat)

        # (f) Conv 3×3 valid → 7 × 7  (keeps channels)
        self.conv3 = Conv_BN_GeLU(256, 256,
                                k_size=(3,3),
                                strd=(1,1),
                                pad=(0,0))             # → 7 × 7

        # (g) MaxPool 2×2, stride 1 → 6 × 6
        self.maxp = nn.MaxPool2d(kernel_size=2,
                                 stride=1)              # → 6 × 6

        # (h) Conv 3×3 valid & channel squeeze: 256 → 128
        self.conv4 = Conv_BN_GeLU(256, 128,
                                k_size=(3,3),
                                strd=(1,1),
                                pad=(0,0))             # → 4 × 4

        # (i) Classifier head
        self.flatten = nn.Flatten()                     # → 128·4·4 = 2048
        self.fc1     = nn.Linear(2048, 256, bias=True)
        self.relu1   = nn.ReLU(inplace=True)
        self.drop1   = nn.Dropout(p=dropout1)
        
        self.fc2     = nn.Linear(256, 6)                # six emotion classes
        self.drop2   = nn.Dropout(p=dropout2)


    # ----------------- FORWARD PASS --------------------------
    def forward(self, x):
        x  = self.conv1(x)         # (B,32,40,109)

        # context branches
        a  = self.conv2_1(x)             # (B,32,18,53)
        b  = self.conv_dil2_2(x)             # (B,32,18,53)
        x  = torch.cat((a, b), dim=1)    # (B,64,18,53)

        x  = self.conv_fuse(x)           # (B,256,18,27)
        x  = self.avgp(x)                 # (B,256, 9, 9)

        x  = self.cbam(x)                # (B,256, 9, 9)

        x  = self.conv3(x)               # (B,256, 7, 7)
        x  = self.maxp(x)                # (B,256, 6, 6)

        x  = self.conv4(x)               # (B,128, 4, 4)

        x  = self.flatten(x)             # (B,2048)
        x  = self.drop1(self.relu1(self.fc1(x)))  # (B,256)
        logits = self.drop2((self.fc2(x)))             # (B,  6)

        return logits

###  ------------ RANDOM SEED ------------------------------
def set_torch_seed(seed: int, threads: int = 1):
    """
    Sets random seeds for Python, NumPy, and PyTorch to guarantee reproducibility.
    Also forces PyTorch to use a fixed number of CPU threads (if needed).

    Parameters:
    -----------
    seed : int
        Seed for random generators.
    threads : int
        Number of CPU threads to use (optional).
    """
    # 1. Python built-in random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (CPU)
    torch.manual_seed(seed)

    # 4. PyTorch (CUDA, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. Make CuDNN deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Set PyTorch threads (optional)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)

###  ------------  XAVIER (GLOROT) UNIFORM Initialization  ------------------------------
def xavier_init(m):
    """
    Apply Xavier uniform initialization to weight parameters and
    zero initialization to bias, for supported layer types.

    Parameters:
    -----------
    m : nn.Module
        A submodule of the model. If it is one of the supported types,
        initialize its weight and bias accordingly.
    """
    # List of layer classes we want to initialize
    types = (
        nn.Linear,
        nn.Conv2d
    )
    if isinstance(m, types):
        # 1. Xavier uniform on weight
        #    torch.nn.init.xavier_uniform_(tensor, gain=1.0)
        #    - Origin: torch.nn.init (PyTorch core library)
        #    - Argument `gain` is a scaling factor (default=1.0).
        nn.init.xavier_uniform_(m.weight, gain=1.0)

        # 2. Zeros on bias (if it exists)
        if m.bias is not None:
            # torch.nn.init.zeros_(tensor)
            nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────
###  MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    set_torch_seed(args.seed, args.threads)

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    ### ----------- DataLoaders ---------------------
    train_dl, dev_dl, test_dl = create_dataloaders(args.batch_size)

    ### ----------- MODEL ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN(dropout1=args.drop1, dropout2=args.drop2, red_rat=args.red_rat).to(device=device) # storing model on cuda

    # Applying Xavier Initialization 
    model.apply(xavier_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ### ----------- TRAINING LOOP ---------------------
    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        train_correct = 0
        train_total = 0
        for feats, label in train_dl:
            feats = feats.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(feats)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            pred_train = out.argmax(dim=1)
            train_correct += (pred_train == label).sum().item()
            train_total += label.size(0)
        
        train_losses.append(epoch_loss / batches)
        train_acc = train_correct / train_total if train_total else 0

        # Quick dev accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, label in dev_dl:
                feats = feats.to(device)
                label = label.to(device)
                pred = model(feats).argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        acc = correct / total if total else 0
        print(f"Epoch {epoch + 1}: train loss {train_losses[-1]:.4f}, train accuracy {train_acc:.4f}, dev accuracy {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.logdir, "model.pt"))

    ### ----------- FINAL EVALUATION ---------------------
    # Generate test set predictions
    model.eval()
    test_predictions = []

    with torch.no_grad():
        for feats, _ in test_dl:  # Assuming no labels in test set
            feats = feats.to(device)
            outputs = model(feats)
            preds = outputs.argmax(dim=1).cpu().numpy()
            test_predictions.extend(preds)
            # If you have filenames: test_filenames.extend(batch_filenames)

    # Save predictions to file

    prediction_file = os.path.join(args.logdir, "predictions.csv")
    with open(prediction_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prediction'])  # Header
        for i, pred in enumerate(test_predictions):
            # If you have filenames: writer.writerow([test_filenames[i], pred])
            writer.writerow([f"sample_{i}", pred])

    print(f"Predictions saved to {prediction_file}")


# ─────────────────────────────────────────────────────────────
###  PREDICTION FILE
# ─────────────────────────────────────────────────────────────




if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
