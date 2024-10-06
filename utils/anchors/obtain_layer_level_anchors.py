import torch

anchors_dir = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings"
anchor_prefix = "llama2-13b_mistral-7b_200000anchors_seed1"
anchor_path = f"{anchors_dir}/{anchor_prefix}.pt"
models = ["llama2-13b", "mistral-7b"]
layer_pair = [40, 32]

output_path = f"{anchors_dir}/{anchor_prefix}_layer{'-'.join([str(i) for i in layer_pair])}.pt"


state = torch.load(anchor_path)

for index, model in enumerate(models):
    state[model] = state[model][layer_pair[index]]

torch.save(state, output_path)