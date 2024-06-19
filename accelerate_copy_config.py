import os
import shutil

with open('accelerate_linux_path.txt', 'r', encoding='utf-8') as f:
    accelerate_linux_path = f.read().strip()

new_path = './accelerate_config_dual_gpu.yaml'

shutil.copy(accelerate_linux_path, new_path)
