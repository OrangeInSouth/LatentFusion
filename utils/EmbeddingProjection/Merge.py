import torch
import pdb

common_path = "/data7/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection"
models = ["EstimationEmbeddingProjection.pt", "EstimationEmbeddingProjection_160000anchors_seed1.pt"]
output_path = common_path + "/Merged_EstimationEmbeddingProjection.pt"

def merge1(weights_list):
    pdb.set_trace()
    weights = torch.stack(weights_list)
    weights = weights.mean(dim=0)
    return weights


states = []
for model in models:
    states.append(torch.load(f"{common_path}/{model}"))

final_state = states[0]
final_state["transformation_weights"]["M"] = merge1([state["transformation_weights"]["M"] for state in states])
torch.save(final_state, output_path)