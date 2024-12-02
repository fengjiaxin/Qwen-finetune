import os
import httpx
from configs.model_config import LLM_DEVICE, MODEL_PATH, LLM_MODELS
from configs.server_config import FSCHAT_MODEL_WORKERS
from configs.server_config import HTTPX_DEFAULT_TIMEOUT
from typing import (Union,
                    Dict)


def get_model_worker_config(model_name: str = None) -> dict:
    '''
    加载model worker的配置项。
    优先级:FSCHAT_MODEL_WORKERS[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    # 本地模型
    llm_model_path = MODEL_PATH[model_name]
    config["model_path"] = llm_model_path
    if llm_model_path and os.path.isdir(llm_model_path):
        config["model_path_exists"] = True
    config["device"] = LLM_DEVICE
    return config


def fschat_controller_address() -> str:
    from configs.server_config import FSCHAT_CONTROLLER

    host = FSCHAT_CONTROLLER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = LLM_MODELS[0]) -> str:
    if model := get_model_worker_config(model_name):
        host = model["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def fschat_openai_api_address() -> str:
    from configs.server_config import FSCHAT_OPENAI_API

    host = FSCHAT_OPENAI_API["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"


def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT
):
    '''
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    '''
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout


def get_httpx_client(
        use_async: bool = False,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    helper to get httpx client
    '''
    # construct Client
    kwargs.update(timeout=timeout)
    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)
