import torch
import pdb
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.EmbeddingProjection.EmbeddingProjection import DeepEmbeddingProjection, EstimationEmbeddingProjection

embedding_projection_path = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/embedding_projection/"\
    "EstimationEmbeddingProjection_200000anchors_seed1.pt"
anchor_embeddings_path = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/llama2-13b_mistral-7b_200000anchors_seed1_layer40-32.pt"
output_path = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/llama2-13b_mistral-7b_filtered100000anchors_seed1_layer40-32.pt"
filtered_anchor_num = 100000
src_model = "mistral-7b"
tgt_model = "llama2-13b"
seed = 1
device = "cuda:0"

torch.manual_seed(seed)

# 1. Load Embedding Projection
embedding_projection = EstimationEmbeddingProjection()
embedding_projection.load(embedding_projection_path, device=device)

# 2. Load Raw Anchor Embeddings
state = torch.load(anchor_embeddings_path)
# src_embeddings = state["mistral-7b"][32]
# tgt_embeddings = state["llama2-13b"][40]
src_embeddings = state[src_model]
tgt_embeddings = state[tgt_model]

# 1. split the dataset
dataset = TensorDataset(src_embeddings, tgt_embeddings)
train_size = len(dataset) - 4000
val_size = 2000
test_size = 2000

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 3. Score Anchors
scores = []
index = 0
batch_num = 10000
while index < len(train_dataset):
    batch_data = train_dataset[index:index + batch_num]
    batch_src_embeds = batch_data[0].to(device)
    batch_tgt_embeds = batch_data[1].to(device)
    pred_embeds = embedding_projection.transform(batch_src_embeds)
    batch_scores = (pred_embeds - batch_tgt_embeds).abs().mean(dim=-1)
    scores.append(batch_scores)
    index += batch_num

scores = torch.concatenate(scores, dim=0)

# pdb.set_trace()
# 4. Filter Anchors
res_sample_indices = scores.topk(filtered_anchor_num)[1]
res_anchors = train_dataset[res_sample_indices.tolist()]
res_src_embeddings = res_anchors[0]
res_tgt_embeddings = res_anchors[1]

state[src_model] = res_src_embeddings
state[tgt_model] = res_tgt_embeddings

# 5. Save Filtered Anchors
torch.save(state, output_path)
print(f"Filterd Anchor Embedding Saved: {output_path}")