# Qwen-finetune
基于千问大模型进行微调的工程项目

参考 
https://blog.csdn.net/m0_50972200/article/details/140667653
https://www.nowcoder.com/discuss/622382524571963392


- finetune.py 微调模型的python脚本
- finetune_lora_single_gpu.sh 微调模型的shell脚本
- process_data_law.py 将数据调整为训练需要的格式
- merge_lora_model.py 合并lora模型
- web_demo.py 通过web的服务方式显示大模型


webShow目录：将启动模型和界面显示写在一个py中
- config.py : 配置相关信息
- startup.py : 启动文件
- webui.py : 界面显示

启动
python webShow/startup.py


webServer目录：启动一个后端服务，在启动一个前端服务
- flowServer.py : 启动后端服务
- testClient.py: 测试后端是否提供服务
- webui.py: 前端服务
- startup.py : 启动前端服务
python webServer/startup.py

