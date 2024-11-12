from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer

model_path = r"D:\models\Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

prompt = "你好，有什么可以帮助你的"

messages = [{"role": "system", "content": '你是中医医疗问答助手章鱼哥，你将帮助用户解答中医相关的医疗问题。'},
            {"role": "user", "content": prompt}
            ]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  #参数用于在输入中添加生成提示，该提示指向 <|im_start|>assistant\n
)

print(text)

# <|im_start|>system
# 你是中医医疗问答助手章鱼哥，你将帮助用户解答中医相关的医疗问题。<|im_end|>
# <|im_start|>user
# 你好，有什么可以帮助你的<|im_end|>
# <|im_start|>assistant

max_length = tokenizer.model_max_length
print(max_length)
