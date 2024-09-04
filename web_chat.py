import json
import sys
from threading import Thread
from queue import Queue

import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import time

if torch.cuda.is_available():
    device = "auto"
else:
    device = "cpu"


def reformat_text(input_text):
    prompt = f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项列表，请输出文本内容的正确分类<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def reformat_sft(instruction, input):
    if input:
        prefix = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
    else:
        prefix = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    prefix = prefix.replace("{instruction}", instruction)
    prefix = prefix.replace("{input}", input)
    return prefix



class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        # self.text_queue = []
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            word = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens)
            # self.text_queue.append(word)
            self.text_queue.put(word)

    def end(self):
        # self.text_queue.append(None)
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


def main(
        base_model: str = "",
        lora_weights: str = "",
        share_gradio: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device,
        trust_remote_code=True,
        # torch_dtype=torch.float16
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights
        )

    model.eval()

    def evaluate(
            input,
            temperature=0.1,
            max_new_tokens=128,
            **kwargs,
    ):
        print(input,
              temperature,
              max_new_tokens,
              **kwargs)
        if not input:
            return

        prompt = reformat_sft(input, "")
        # prompt = reformat_text(input)
        # prompt = input

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        if not (1 > temperature > 0):
            temperature = 1
        if not (2000 > max_new_tokens > 0):
            max_new_tokens = 200

        output = ''

        streamer = TextIterStreamer(tokenizer)
        generation_config = dict(
            temperature=temperature,
            top_p=1,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            repetition_penalty=1.2,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        c = Thread(target=lambda: model.generate(input_ids=input_ids, **generation_config))
        c.start()
        for text in streamer:
            output = text
            yield output
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(output)

    with gr.Blocks() as demo:
        gr.Markdown(
            "# 测试模型")
        with gr.Row():
            with gr.Column():  # 列排列
                context = gr.Textbox(
                    lines=3,
                    label="Instruction",
                    placeholder="输入问题 ..",
                )
                temperature = gr.Slider(
                    minimum=0, maximum=1, value=0.3, label="Temperature(温度越高，越随机)"
                )
                max_tokens = gr.Slider(
                    minimum=1, maximum=2000, step=1, value=300, label="Max tokens(输出最多token数量)"
                )
            with gr.Column():
                answer1 = gr.Textbox(
                    lines=1,
                    label="回答",
                )
        with gr.Row():
            submit = gr.Button("submit", variant="primary")
            gr.ClearButton([context, answer1])
        submit.click(fn=evaluate, inputs=[context, temperature, max_tokens],
                     outputs=[answer1])

    demo.queue().launch(server_name="localhost", share=share_gradio)
    # Old testing code follows.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='云起无垠SecGPT模型RLHF测试')
    parser.add_argument("--base_model", type=str, required=True, help="基础模型")
    parser.add_argument("--lora", type=str, help="lora模型")
    parser.add_argument("--share_gradio", type=bool, default=False, help="开放外网访问")
    args = parser.parse_args()
    main(args.base_model, args.lora, args.share_gradio)
