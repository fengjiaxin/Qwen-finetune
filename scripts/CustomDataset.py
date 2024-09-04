from torch.utils.data import Dataset
import json
from utils import format_example, apply_template


class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, device, max_length):
        self.data = json.load(open(data_path,encoding="utf-8"))
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        formatted_example = format_example(example)
        context = formatted_example['context']
        target = formatted_example["target"]
        inputs = self.tokenizer(
            apply_template(context),
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        labels = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs['labels'] = labels['input_ids']
        # 确保所有张量在同一个设备上
        return {key: val.squeeze().to(self.device) for key, val in inputs.items()}

