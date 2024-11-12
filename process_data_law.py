import json



# 读取以.jsonl结尾的文件

json_data = []

with open('./data/DISC-Law-SFT-Triplet-released.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        json_data.append(data)

template = []

for idx, data in enumerate(json_data[:]):
    conversation = [
        {
            "from":"user",
            "value":data["input"]
        },
        {
            "from":"assistant",
            "value":data["output"]
        }
    ]
    template.append({
        "id":f"identity_{idx}",
        "conversations":conversation
    })
print(len(template))
print(json.dumps(template[2], ensure_ascii=False, indent=2))

output_file_path = "./data/train_data_law.json"
with open(output_file_path,"w", encoding="utf-8") as f:
    json.dump(template, f, ensure_ascii=False, indent=2)
print("done")
