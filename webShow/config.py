import torch
import sys

# ====大模型相关配置
MODEL_PATH = r"D:\models\Qwen2.5-0.5B-Instruct"
LORA_PATH = r""
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
USE_LORA = False


# web服务相关配置
# 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
PORT=8051