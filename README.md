# Qwen-finetune
基于千问大模型进行微调的工程项目

参考 
https://blog.csdn.net/m0_50972200/article/details/140667653

scripts 目录：微调模型的相关代码
- config.py  配置文件
- CustomDataset.py 处理数据
- utils.py  提供一些工具类
- Qwen2-ft.py 训练脚本

merge_lora_model.py 合并lora模型
web_chat.py web端调用大模型回答
train.sh 训练脚本