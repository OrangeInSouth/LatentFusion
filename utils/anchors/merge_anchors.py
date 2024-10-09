import torch
import os
from tqdm import tqdm

anchor_dir = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/"
anchor_prefix = "llama2-13b_mistral-7b_1000000anchors_seed1"
output_path = f"/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/{anchor_prefix}.pt"

files = [f for f in os.listdir(anchor_dir) if f.startswith(anchor_prefix) and f != anchor_prefix + '.pt']
files = sorted(files)

states = []
for file in tqdm(files):
    state = torch.load(f"{anchor_dir}/{file}", map_location=torch.device('cpu'))
    states.append(state)

models = list(states[0].keys())

res_state = {}
for model in models:
    layer_num = len(states[0][model])
    model_state = [torch.concatenate([state[model][l] for state in states]) for l in range(layer_num)]
    res_state[model] = model_state


torch.save(res_state, output_path)

print(f"Merged {len(res_state[models[0]][0])} anchors.")
print(f"Saved into: {output_path}")
