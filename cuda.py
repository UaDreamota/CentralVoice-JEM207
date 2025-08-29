import torch, platform
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)  # None => CPU-only build
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
