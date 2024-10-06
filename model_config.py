import os
model_dir = "/share/home/fengxiaocheng/baohangli/ModelsHub/"

model_paths = {"mistral-7b": "mistralai/Mistral-7B-v0.1",
        "internlm-20b": "internlm-20b",
        "skywork-13b": "Skywork/Skywork-13B-base",
        "llama2-13b": "Llama-2-13b-hf",
        "yi-6b": "01-ai/Yi-6B-hf",
        "tigerbot2-13b": "TigerResearch/tigerbot-13b-base-v2/",
        "nanbeige-16b": "Nanbeige/Nanbeige-16B-Base"}

for model, model_path in model_paths.items():
    model_paths[model] = os.path.join(model_dir, model_path)

