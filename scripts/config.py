import torch

# 最大token长度
MAX_TOKEN_LENGTH = 512

# batch size 大小
BATCH_SIZE = 32

# EPOCH
EPOCH = 50

# 判断设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 预训练模型地址
MODEL_PATH = r"D:\workspace\medical-ft\models\Qwen2-0.5B-Instruct"

# 预训练模型地址
DATA_PATH = r"D:\workspace\medical-ft\data\cpmi_dataset_100.json"

# 训练过程中保存的模型
OUTPUT_DIR = r"D:\workspace\medical-ft\output"

LORA_DIR = r"D:\workspace\medical-ft\lora"
