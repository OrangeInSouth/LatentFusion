import torch
import os
from tqdm import tqdm
import pdb
import re
import argparse

anchor_dir = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/"
anchor_prefix = "llama2-13b_mistral-7b_1000000anchors_seed1"

def parse_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")

    # Add arguments with default values
    parser.add_argument('--models', nargs='+', default=["llama2-13b", "mistral-7b"], help="models, list of strings")
    parser.add_argument('--layer-pair', nargs='+', default=[40, 32], help="models, list of strings")

    # Parse the arguments
    args = parser.parse_args()
    return args

args = parse_args()

models = args.models
layer_pair = [int(i) for i in args.layer_pair]

output_path = f"{anchor_dir}/{anchor_prefix}_layer{'-'.join([str(i) for i in layer_pair])}.pt"

files = [f for f in os.listdir(anchor_dir) if re.match(fr"^{re.escape(anchor_prefix)}_\d+\.pt$", f)]
files = sorted(files)

states = []
for file in tqdm(files):
    state = torch.load(f"{anchor_dir}/{file}", map_location=torch.device('cpu'))
    single_layer_state = {}
    for index, model in enumerate(models):
        try:
            assert len(state[model][layer_pair[index]].shape) == 2
            single_layer_state[model] = state[model][layer_pair[index]]
        except Exception:
            pdb.set_trace()
    states.append(single_layer_state)

res_state = {}
for model in models:
    res_state[model] = torch.concatenate([state[model] for state in states])


torch.save(res_state, output_path)

print(f"Merged {len(res_state[models[0]])} anchors.")
print(f"Saved into: {output_path}")
