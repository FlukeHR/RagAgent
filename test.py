import torch
import bitsandbytes as bnb

# 检查能否在 GPU 上创建 8-bit 线性层
try:
    linear8bit = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False, threshold=6.0)
    linear8bit.cuda()  # 尝试移到 GPU
    print("✅ bitsandbytes + CUDA is working!")
except Exception as e:
    print("❌ bitsandbytes CUDA not available:", e)