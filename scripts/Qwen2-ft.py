# Qwen2-ft.py
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, Trainer
import os
from utils import prepare_model, print_model_parameters, save_model
from config import (
    DEVICE,
    MODEL_PATH,
    DATA_PATH,
    OUTPUT_DIR,
    LORA_DIR,
    MAX_TOKEN_LENGTH,
    BATCH_SIZE,
    EPOCH
)
from CustomDataset import CustomDataset

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
model = prepare_model(MODEL_PATH)
model = model.to(DEVICE)  # 把模型移到设备上

print_model_parameters(model)

# 配置并应用 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型 (Causal LM)
    inference_mode=False,  # 模式：训练模式 (inference_mode=False)
    r=8,  # 低秩分解的秩：8 (r=8)
    lora_alpha=32,  # 缩放因子：32 (lora_alpha=32)
    lora_dropout=0.1,  # dropout 概率：0.1 (lora_dropout=0.1)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 查询投影和值投影模块
)
model = get_peft_model(model, peft_config)  # 应用LoRA配置

# 准备数据集
train_dataset = CustomDataset(DATA_PATH, tokenizer, DEVICE, MAX_TOKEN_LENGTH)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # 训练结果保存的目录
    num_train_epochs=EPOCH,  # 训练的总轮数
    per_device_train_batch_size=BATCH_SIZE,  # 每个设备上的训练批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数，在进行反向传播前累积多少步
    logging_steps=100,
    save_strategy="epoch",  # 保存策略，每个 epoch 保存一次模型
    learning_rate=5e-5,  # 学习率
    fp16=True,  # 启用 16 位浮点数训练，提高训练速度并减少显存使用
    dataloader_pin_memory=False,  # 禁用 pin_memory 以节省内存
)

# 定义 Trainer
trainer = Trainer(
    model=model,  # 训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
)

# 创建保存模型的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# 开始训练
trainer.train()
save_model(model,LORA_DIR)

print(f"LORA Model saved to {LORA_DIR}")
