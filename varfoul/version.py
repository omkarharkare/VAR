##<--- Jainil --->##
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")