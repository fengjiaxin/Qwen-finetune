# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread
from peft import PeftModel
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = '/workspace/models/Qwen2-7B-Instruct'
LORA_PATH = "/workspace/qwen-main/output/qwen2-7B-Instruct"
 
def _get_args():
    parser = ArgumentParser(description="Qwen2-Instruct web chat demo.")

    parser.add_argument("--checkpoint-path", type=str, default=MODEL_PATH,
                        help="Checkpoint model path, default to %(default)r")

    parser.add_argument("--lora_path", type=str, default=LORA_PATH,
                        help="Checkpoint lora path, default to %(default)r")

    parser.add_argument("--use_lora",action="store_true",
                        help="是否使用lora地址，注意lora_path不能为空")

    parser.add_argument("--cpu-only", action="store_true",
                        help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true",
                        help="Create a publicly shareable link for the interface.")

    parser.add_argument("--inbrowser", action="store_true",
                        help="Automatically launch the interface in a new tab on the default browser.")

    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")

    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        resume_download=False,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda" if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        resume_download=False,
    ).eval()

    if args.use_lora and args.lora_path:
        # 加载lora权重
        print(args.use_lora)
        model = PeftModel.from_pretrained(model, model_id=LORA_PATH)
    model.generation_config.max_new_tokens = 2048  # For chat.

    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)

            yield _chatbot
            full_response = response

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Qwen: {full_response}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png" style="height: 120px"/><p>""")
        gr.Markdown(
            """\
<center><font size=3>本WebUI基于Qwen2-Instruct打造，实现聊天机器人功能</center>"""
        )

        chatbot = gr.Chatbot(label="Qwen", elem_classes="control-height")
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )


    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()

#python web_demo.py --server-port 8000
#python web_demo.py --use_lora --server-port 8001
