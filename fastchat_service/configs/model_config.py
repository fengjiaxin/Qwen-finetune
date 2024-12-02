import torch


# 要运行的 LLM 名称，只使用本地模型。列表中本地模型将在启动项目时全部加载。
# 列表中第一个模型将作为 API 和 WEBUI 的默认模型。
LLM_MODELS = ["Qwen2-0.5B-Instruct"]

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型名称 -> 绝对路径
MODEL_PATH = {
    "Qwen2-0.5B-Instruct": r"D:\models\Qwen2-0.5B-Instruct"
}
