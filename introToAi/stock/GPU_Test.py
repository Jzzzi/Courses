import torch

# 打印所有可用GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 设置当前设备为GPU0
torch.cuda.set_device(0)