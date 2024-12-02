import time
import requests
from openai import OpenAI


# 测试api相应是否无误
def request_test():
    # 发送POST请求

    url = "http://127.0.0.1:20000/v1/chat/completions"  # API的地址
    data = {

        "model": "Qwen2-0.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "你是谁？"
            }
        ]
    }

    response = requests.post(url, json=data)

    # 解析响应
    if response.status_code == 200:
        answer = response.json()
        choice = answer["choices"][0]
        message = choice["message"]
        content = message["content"]
        print(content)
    else:
        print("Request failed:", response.text)


def open_api_test():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:20000/v1"
    )

    def get_completion(prompt, model="Qwen2-0.5B-Instruct"):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,

        )
        return response.choices[0].message.content

    prompt = "你是谁？"
    print(get_completion(prompt))


if __name__ == '__main__':
    time_start = time.time()
    open_api_test()
    # request_test()
    time_end = time.time()
    print("测试时间 - {:.2f} s".format(time_end - time_start))
