from transformers import AutoModelForCausalLM
import transformers


# example格式 {"instruction":"","input":"","output":""}
def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    target = example["output"]
    return {"context": context, "target": target}


def apply_template(question):
    res = f"<|im_start|>system\n你是中医医疗问答助手章鱼哥，你将帮助用户解答中医相关的医疗问题。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return res


def prepare_model(pretrain_train_path):
    # 加载模型
    mode_config = transformers.AutoConfig.from_pretrained(
        pretrain_train_path,
        trust_remote_code=False
    )
    mode_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pretrain_train_path, trust_remote_code=False, config=mode_config)
    model.supports_gradient_checkpointing = True  # 节约cuda
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def save_model(model, path):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


# 打印模型参数
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params / 1000000}M total:{total_params}')
