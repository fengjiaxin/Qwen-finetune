from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer

data_path = r"D:\workspace\medical-ft\data\cpmi_dataset_100.json"
model_path = r"D:\models\Qwen2-1.5B-Instruct"

data = json.load(open(data_path, encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
max_length = 16
device = "cpu"

test_input = "使用中医知识正确回答适合这个病例的中成药。"
test_target = "我前几天吃了很多食物，但肚子总是不舒服，咕咕响，还经常嗳气反酸，大便不成形，脸色也差极了。"

inputs = tokenizer(
    test_input,
    max_length=max_length,
    truncation=True,
    padding='max_length',
    return_tensors="pt"
)

labels = tokenizer(
    test_target,
    max_length=max_length,
    truncation=True,
    padding='max_length',
    return_tensors="pt"
)
inputs['labels'] = labels['input_ids']
print(inputs)
# {'input_ids': tensor([[ 37029, 104823, 100032,  88991, 102104, 100231,  99487, 103095,   9370,
#           15946,  12857,  99471,   1773, 151643, 151643, 151643]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]),
#  'labels': tensor([[ 35946, 112607,  99405, 104686, 102153,   3837,  77288, 105925, 104014,
#          110237,   3837, 113898, 113898,  99365,   3837,  97706]])
# }

print(inputs['input_ids'].squeeze()) # 改变维度
# tensor([ 37029, 104823, 100032,  88991, 102104, 100231,  99487, 103095,   9370,
#          15946,  12857,  99471,   1773, 151643, 151643, 151643])

