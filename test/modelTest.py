import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 指定模型
MODEL = r"D:\models\Qwen2-1.5B-Instruct"
LORA_PATH = r"D:\git\Qwen-finetune\lora\cpmi"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=False)
model.to(device)
# 模型设为评估状态
model.eval()

if LORA_PATH:
    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=LORA_PATH)


# 定义测试示例
test_example = {
    "instruction": "使用中医知识正确回答适合这个病例的中成药。",
    "input": "肛门疼痛，痔疮，肛裂。"
}


# 讲指令和问题合并为一句话
def merge(instruction, input):
    prompt = f"Instruction: {instruction}\nInput: {input}"
    return prompt


def apply_template(question):
    res = f"<|im_start|>system\n你是中医医疗问答助手章鱼哥，你将帮助用户解答中医相关的医疗问题。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return res


question = merge(test_example['instruction'], test_example['input'])
context = apply_template(question)

model_inputs = tokenizer(context, return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids,
                               max_length=512,
                               num_return_sequences=1)  # 指定返回概率最高的 topN 个序列
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"提问: {question}")
print(f"回答: {res}\n")
print("===========")


# 提问: Instruction: 使用中医知识正确回答适合这个病例的中成药。
# Input: 肛门疼痛，痔疮，肛裂。
# 回答: 病情描述：患者肛门部位疼痛，伴有便血、脱垂等症状。
#
# 推荐使用：地榆槐角丸（含大黄）或者金匮肾气丸。这两种药物都有清热解毒、收敛止痛的功效，对治疗肛肠疾病有较好的效果。其中，地榆槐角丸适用于急性或慢性细菌性痢疾引起的下痢赤白脓血等；金匮肾气丸则可用于治疗脾虚型慢性盆腔炎引起的带下量多色淡，腰膝酸软，神疲乏力等症。建议在医师指导下服用，并定期复查病情变化。