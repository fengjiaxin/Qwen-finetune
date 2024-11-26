import requests
import json


def dump_output(r):
    print("\n")
    for x in r.iter_content(None, decode_unicode=True):
        line = x.strip()
        print("===================")
        print(line, flush=True)
        left_pos = line.find('{')
        right_pos = line.find('}')
        dict_str = line[left_pos:right_pos + 1]
        print(dict_str)
        dic = json.loads(dict_str)
        print(dic)
        print("++++++++++++++++")


def test_stream_chat():
    data = {
        "query": "请用100字左右的文字介绍自己",
        "max_tokens": 1024,
        "top_p": 1.0,
        "temperature": 0.7
    }

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    url = "http://10.200.101.150:8080/stream"

    response = requests.post(url, headers=headers, json=data, stream=True)
    dump_output(response)
    assert response.status_code == 200


def test_chat():
    url = 'http://10.200.101.150:8080/chat'
    data = {
        'query': '介绍以下你自己'
    }

    response = requests.post(url, data=json.dumps(data))
    res_dic = json.loads(response.text)
    print(res_dic["response"])
    print(response.text)
    # {"query":"介绍以下你自己","response":"我是来自阿里云的大规模语言模型“通义千问”，我是一个能够回答问题、创作文字，还能表达观点、撰写代码的超大规模语言模型。我拥有多项专业能力，例如但不限于撰写公文、创意写作、文本生成、撰写故事、讨论哲学、提供数据分析、编写代码等。我作为一个语言模型，能够生成流畅且有逻辑的回复，以帮助用户解决问题、提供见解或建议。在与我交流时，您可以提出各种问题或任务，我将竭尽所能地提供最合适的答案。期待与您进行有意义的对话！\n","success":true}


if __name__ == "__main__":
    test_stream_chat()
