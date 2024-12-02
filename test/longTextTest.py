import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from document_loaders import *
from string import Template


#长文本解读能力测试

class LLMHelper:
    def __init__(self, model_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False)
        self.model.to(self.device)
        # 模型设为评估状态
        self.model.eval()

    def chat(self,
             prompt,
             max_tokens=1024,
             top_p=1.0,
             temperature=0.8
             ):
        conversation = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(inputs.input_ids, max_new_tokens=max_tokens, top_p=top_p,
                                            temperature=temperature)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    # MODEL = r"D:\models\Qwen2-1.5B-Instruct"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # llmHelper = LLMHelper(MODEL, device)

    doc_path = r"D:\git\Qwen-finetune\test\files\大模型功能设计.docx"
    loader = RapidOCRDocLoader(file_path=doc_path)
    doc = loader.load()[0]
    content = doc.page_content
    print(content)
    print(len(content))

    # question = "模型功能有哪些，列举出来"
    #
    # instruct_str = "请参考一下内容\n\n${text}\n\n，回答问题\n${question}"
    # template = Template(instruct_str)
    # prompt = template.substitute(text=doc.page_content, question=question)
    #
    # res = llmHelper.chat(prompt)
    # print(res)
