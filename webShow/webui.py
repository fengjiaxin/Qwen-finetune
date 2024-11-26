from peft import PeftModel
from threading import Thread
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from config import (MODEL_PATH, LORA_PATH, DEVICE, USE_LORA)


class LLMHelper:
    def __init__(self, model_path, lora_path, use_lora, device_map):
        self.model_path = model_path
        self.lora_path = lora_path
        self.use_lora = use_lora
        self.device_map = device_map
        # 加载模型
        self._load_model()

    def _load_model(self):
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                resume_download=False))
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map=self.device_map,
                resume_download=False).eval())
        if self.use_lora and self.lora_path:
            model = PeftModel.from_pretrained(model, model_id=self.lora_path)

        self.model = model

    def chat_stream(self,
                    query,
                    history,
                    history_len=3,
                    max_length=1024,
                    top_p=1.0,
                    temperature=0.8
                    ):
        conversation = history[-history_len * 2:]
        conversation.append({"role": "user", "content": query})
        input_text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_length": max_length,
            "top_p": top_p,
            "temperature": temperature
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text


if __name__ == "__main__":
    llmHelper = LLMHelper(MODEL_PATH, LORA_PATH, USE_LORA, DEVICE)

    st.set_page_config(
        page_title="Streamlit Simple Demo",
        page_icon=":robot:",
        layout="wide"
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    history_len = st.sidebar.slider("history_len", 1, 10, 3, step=1)
    max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
    top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

    buttonClean = st.sidebar.button("清理会话历史", key="clean")
    if buttonClean:
        st.session_state.history = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()

    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            with st.chat_message(name="user", avatar="user"):
                st.markdown(message["content"])
        else:
            with st.chat_message(name="assistant", avatar="assistant"):
                st.markdown(message["content"])

    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty()

    prompt_text = st.chat_input("请输入您的问题")
    if prompt_text:
        input_placeholder.markdown(prompt_text)
        history = st.session_state.history
        res_text = ""
        for rs in llmHelper.chat_stream(
                prompt_text,
                history,
                history_len=history_len,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature):
            res_text += rs
            message_placeholder.markdown(res_text)
        st.session_state.history.append({"role": "user", "content": prompt_text})
        st.session_state.history.append({"role": "assistant", "content": res_text})
