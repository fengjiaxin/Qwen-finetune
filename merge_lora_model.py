import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 应用场景
# 1.在baseModel(Qwen2-0.5B)的基础上，通过lora的预训练方式微调模型，保存的lora地为lora_path
# 2.目标，合并模型，方便后续使用

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def apply_lora(model_path, output_path, lora_path):
    print(f"Loading the base model from {model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, device_map=DEVICE, torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    lora_path = r"D:\workspace\medical-ft\lora"
    model_path = r"D:\workspace\medical-ft\models\Qwen2-0.5B-Instruct"
    output = r"D:\workspace\medical-ft\result\Qwen2-0.5B-Instruct-lora"

    apply_lora(model_path, output, lora_path)
