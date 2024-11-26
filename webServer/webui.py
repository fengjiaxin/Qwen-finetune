import streamlit as st
import torch
import requests
import json
import argparse


def handle_line(line):
    line = line.strip()
    left_pos = line.find('{')
    right_pos = line.find('}')
    dict_str = line[left_pos:right_pos + 1]
    dic = json.loads(dict_str)
    return dic


def main(url):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    st.set_page_config(
        page_title="Streamlit Simple Demo",
        page_icon=":robot:",
        layout="wide"
    )

    max_tokens = st.sidebar.slider("max_tokens", 0, 8192, 2048, step=1)
    top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

    buttonClean = st.sidebar.button("清理会话历史", key="clean")
    if buttonClean:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()

    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty()

    prompt_text = st.chat_input("请输入您的问题")
    if prompt_text:
        data = {
            'query': prompt_text,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'temperature': temperature,
        }
        input_placeholder.markdown(prompt_text)
        response = requests.post(url, headers=headers, json=data, stream=True)
        for x in response.iter_content(None, decode_unicode=True):
            dic = handle_line(x)
            res_txt = dic["response"]
            message_placeholder.markdown(res_txt)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Streamlit App with Command Line Arguments")

    # 添加命令行参数
    parser.add_argument("--url", type=str, help="User's name", default="John Doe")
    args = parser.parse_args()
    main(args.url)
