
# ─────────────────────────────────────────────────────────────
### IMPORTS
# ─────────────────────────────────────────────────────────────

import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torchmetrics

# ─────────────────────────────────────────────────────────────
### DYNAMIC ARGUMENTS
# ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.01, type = float, help = "Learning rate")
parser.add_argument("--label_smoothing", default=0.05, type = float, help = "Label smoothing")
parser.add_argument("--dropout_rate", default=0.2, type = float, help = "Dropout rate")
parser.add_argument("--drop_connect_rate", default=0.2, type = float, help = "Dropout connection rate")

# ─────────────────────────────────────────────────────────────
### DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
### CREMA-D DATASET
# ─────────────────────────────────────────────────────────────

class ModelNetDatataset(torch.utils.data.Dataset):
    def __init__(self, dataset, train_transform=None, eval_transform=None):
        super().__init__(dataset=dataset)

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ...
        
        # Return simple tuple of tensors (this matches the TrainableModule expectations)
        return 
    
##### ───────────────────────────────────────────────────────────── FULLY CONVOLUTIONAL NETWORK WITH CBAM ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
###   1. CONV → BN → GELU BLOCK
# ─────────────────────────────────────────────────────────────
class ConvBNGeLU(nn.Module):
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
 
        return self.act(self.bn(self.conv(x)))

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
    def __init__(self):
        super().__init__()

        # (a) Shrink x-axis from 218 → 109
        self.conv1 = ConvBNGeLU(1, 32,
                                      k_size=(3,3),
                                      strd=(1,2),
                                      pad=(1,1))

        # (b) Two parallel 5×5 context CBs, stride (2,2)
        self.conv2_1 = ConvBNGeLU(32, 32,
                                  k_size=(5,5),
                                  strd=(2,2),
                                  pad=(0,1))           # → 18 × 53

        self.conv_dil2_2 = ConvBNGeLU(32, 32,
                                  k_size=(5,5),
                                  strd=(2,2),
                                  pad=(2,3),            # careful: keeps H,W
                                  dilation=2)           #     18 × 53

        # (c) Fuse & halve x-axis again: 53 → 27
        self.conv_fuse = ConvBNGeLU(64, 256,
                                    k_size=(3,3),
                                    strd=(1,2),
                                    pad=(1,1))          # → 18 × 27

        # (d) 2×3 average pool to 9 × 9
        self.avgp = nn.AvgPool2d(kernel_size=(2,3),
                                stride=(2,3))            # → 9 × 9
        
        # (e) CBAM attention (learned re-weighting)
        self.cbam = CBAM(channels=256)

        # (f) Conv 3×3 valid → 7 × 7  (keeps channels)
        self.conv3 = ConvBNGeLU(256, 256,
                                k_size=(3,3),
                                strd=(1,1),
                                pad=(0,0))             # → 7 × 7

        # (g) MaxPool 2×2, stride 1 → 6 × 6
        self.maxp = nn.MaxPool2d(kernel_size=2,
                                 stride=1)              # → 6 × 6

        # (h) Conv 3×3 valid & channel squeeze: 256 → 128
        self.conv4 = ConvBNGeLU(256, 128,
                                k_size=(3,3),
                                strd=(1,1),
                                pad=(0,0))             # → 4 × 4

        # (i) Classifier head
        self.flatten = nn.Flatten()                     # → 128·4·4 = 2048
        self.fc1     = nn.Linear(2048, 256, bias=True)
        self.relu1   = nn.ReLU(inplace=True)
        self.drop1   = nn.Dropout(p=0.4)
        self.fc2     = nn.Linear(256, 6)                # six emotion classes



    # ----------------- FORWARD PASS --------------------------
    def forward(self, x):
        x  = self.conv_shrink(x)         # (B,32,40,109)

        # context branches
        a  = self.ctx_std(x)             # (B,32,18,53)
        b  = self.ctx_dil(x)             # (B,32,18,53)
        x  = torch.cat((a, b), dim=1)    # (B,64,18,53)

        x  = self.conv_fuse(x)           # (B,256,18,27)
        x  = self.avg(x)                 # (B,256, 9, 9)

        x  = self.cbam(x)                # (B,256, 9, 9)

        x  = self.conv3(x)               # (B,256, 7, 7)
        x  = self.maxp(x)                # (B,256, 6, 6)

        x  = self.conv4(x)               # (B,128, 4, 4)

        x  = self.flatten(x)             # (B,2048)
        x  = self.drop1(self.relu1(self.fc1(x)))  # (B,256)
        logits = self.fc2(x)             # (B,  6)

        return logits

###  ------------ ROUND FILTERS ------------------------------



#------------------------------------------------------------
### MAIN function
#------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    
    modelnet = ModelNet(args.modelnet)

    ### ----------- DataLoaders ---------------------
    train = ModelNetDatataset(modelnet.train, train_transform=train_transform).dataloader(batch_size=args.batch_size, shuffle=True, pin_memory = True)

    dev = ModelNetDatataset(modelnet.dev, eval_transform=eval_transform).dataloader(batch_size=args.batch_size, pin_memory = True)

    test = ModelNetDatataset(modelnet.test, eval_transform=eval_transform).dataloader(batch_size=args.batch_size, pin_memory = True)

    ### ----------- MODEL ---------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an EfficientNet-B0 (base model)
    model = EfficientNet3D(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=args.dropout_rate,
        num_classes=modelnet.LABELS,
        drop_connect_rate=args.drop_connect_rate 
    ).to(device=device)

    ### ----------- MODEL training ---------------------
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs * len(train),
        eta_min=args.lr * 0.1
        )
    ## CONFIGURATION OF THE MODEL
    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=modelnet.LABELS)
        },
        logdir=args.logdir
    )

    ## STORING BEST MODEL
    callbacks = [
        Checkpoint(monitor="dev_accuracy", mode="max")
    ]

    ## .fit
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=callbacks)

    ### ----------- FINAL EVALUATION ---------------------
    results = model.evaluate(dev)
    print(f"Final validation performance: {results}")

#------------------------------------------------------------
### GENERATING PREDICTION FILE
#------------------------------------------------------------

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(grid, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
