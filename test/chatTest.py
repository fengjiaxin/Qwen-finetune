from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer

model_path = r"D:\models\Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


query = "地球到月球距离多远？"
messages = [{"role": "user", "content": query}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  #参数用于在输入中添加生成提示，该提示指向 <|im_start|>assistant\n
)

print(text)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# 地球到月球距离多远？<|im_end|>
# <|im_start|>assistant


text1 = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  #参数用于在输入中添加生成提示，该提示指向 <|im_start|>assistant\n
)
print(text1)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# 地球到月球距离多远？<|im_end|>

max_length = tokenizer.model_max_length
print(max_length)
