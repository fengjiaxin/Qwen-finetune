from peft import PeftModel
from fastapi import FastAPI
from fastapi import Body
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import argparse

import json
import sys
from threading import Thread


class ChatQwen:
    def __init__(self, model_path, lora_path, use_lora, device_map):
        self.model_path = model_path
        self.lora_path = lora_path
        self.use_lora = use_lora
        self.device_map = device_map
        # 加载模型
        print("Start initialize model...")
        self._load_model()
        print("finish initialize model")

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

    def _chat_stream(self,
                     query,
                     max_tokens=1024,
                     top_p=1.0,
                     temperature=0.8
                     ):
        conversation = [{"role": "user", "content": query}]
        input_text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, skip_prompt=True, timeout=360.0, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def chat(self,
             query,
             max_tokens=1024,
             top_p=1.0,
             temperature=0.8
             ):
        conversation = [{"role": "user", "content": query}]
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

    def stream(self,
               query,
               max_tokens=1024,
               top_p=1.0,
               temperature=0.8
               ):
        if query is None:
            yield {"query": "", "response": "", "finished": True}
        response = ""
        for new_text in self._chat_stream(query,
                                          max_tokens,
                                          top_p,
                                          temperature):
            if new_text:
                response += new_text
                yield {"delta": new_text, "response": response, "finished": False}
        yield {"delta": "", "response": response, "finished": True}


def start_server(http_address: str, port: int, model_path: str, lora_path: str, use_lora: bool, device: str):
    bot = ChatQwen(model_path, lora_path, use_lora, device)
    app = FastAPI()
    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"]
                       )

    @app.get("/")
    def index():
        return {'message': 'started', 'success': True}

    @app.post("/chat")
    async def answer_question(query: str = Body(..., description="用户输入"),
                               temperature: float = Body(0.7, description="LLM 采样温度,值越大越随机", ge=0.0, le=2.0),
                               top_p: float = Body(0.8, description="如果累计概率已经超过0.95，剩下的token不会被考虑", ge=0.0, le=2.0),
                               max_tokens: Optional[int] = Body(2048,
                                                                description="限制LLM生成Token数量，默认None代表模型最大值"),
                               ):
        result = {"response": "", "success": False}
        try:
            print("Query - {}".format(query))
            response = bot.chat(query, max_tokens, top_p, temperature)
            print("Answer - {}".format(response))
            result = {"response": response,
                      "success": True}
        except Exception as e:
            print("error: -{}".format(e))
        return result

    @app.post("/stream")
    def answer_question_stream(query: str = Body(..., description="用户输入"),
                               temperature: float = Body(0.7, description="LLM 采样温度,值越大越随机", ge=0.0, le=2.0),
                               top_p: float = Body(0.8, description="如果累计概率已经超过0.95，剩下的token不会被考虑", ge=0.0, le=2.0),
                               max_tokens: Optional[int] = Body(None,
                                                                description="限制LLM生成Token数量，默认None代表模型最大值"),
                               ):
        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')

        try:
            print("query - {}".format(query))
            print("max_tokens - {}".format(max_tokens))
            print("top_p - {}".format(top_p))
            print("temperature - {}".format(temperature))
            if max_tokens is None:
                max_tokens = 32768
            return EventSourceResponse(decorate(bot.stream(query, max_tokens, top_p, temperature)))
        except Exception as e:
            print("error: -{}".format(e))
            return EventSourceResponse(decorate(bot.stream(None)))

    print("starting server...")
    uvicorn.run(app=app, host=http_address, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for Qwen2')
    parser.add_argument('--port', '-P', help='port of this service', default=8080)
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用lora地址，注意lora_path不能为空")
    parser.add_argument("--only_cpu", action="store_true",
                        help="是否只使用cpu推理")
    args = parser.parse_args()

    # 模型地址
    MODEL_PATH = "/workspace/models/Qwen2-1.5B-Instruct"
    LORA_PATH = "/workspace/qwen-main/output/keyan/qwen2-1.5B-Instruct"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.only_cpu:
        DEVICE = "cpu"
    # 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
    DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
    start_server(DEFAULT_BIND_HOST, int(args.port), MODEL_PATH, LORA_PATH, args.use_lora, DEVICE)

    #nohup python flowServer.py --port 8080 --only_cpu > ./server_8080.out 2>&1&
    #nohup python flowServer.py --port 8081 --use_lora --only_cpu > ./server_8081.out 2>&1&
