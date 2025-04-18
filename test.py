import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
