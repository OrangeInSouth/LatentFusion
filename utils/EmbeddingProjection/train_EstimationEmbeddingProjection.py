import sys
import argparse
# sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import time
import pdb
from utils import compute_matrix_scale
import random
from utils.distribution_utils import print_histogram, compare_histogram

from utils.EmbeddingProjection.EmbeddingProjection import DeepEmbeddingProjection, EstimationEmbeddingProjection


      
def parse_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")

    # Add arguments with default values
    parser.add_argument('--anchor-num', type=int, default=200000, help='Number of Anchors used to estimation the embedding projection.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    # Parse the arguments
    args = parser.parse_args()
    return args

# Example usage
# anchor_embeddings_path = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/llama2-13b_mistral-7b_20000anchors_seed1.pt"

if __name__ == "__main__":
    args = parse_args()
    anchor_num = args.anchor_num
    seed = args.seed

    torch.manual_seed(seed)

    anchor_embeddings_path = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/anchor_embeddings/llama2-13b_mistral-7b_200000anchors_seed1_layer40-32.pt"
    save_dir = "/share/home/fengxiaocheng/ychuang/LatentFusion/experiments/embedding_projection"
    state = torch.load(anchor_embeddings_path)
    # src_embeddings = state["mistral-7b"][32]
    # tgt_embeddings = state["llama2-13b"][40]
    src_embeddings = state["mistral-7b"]
    tgt_embeddings = state["llama2-13b"]

    # 1. split the dataset
    dataset = TensorDataset(src_embeddings, tgt_embeddings)
    train_size = len(dataset) - 4000
    val_size = 2000
    test_size = 2000
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 2. training
    print("Estimation:")
    esitmation_embedding_projection = EstimationEmbeddingProjection()
    esitmation_embedding_projection.fit(train_dataset, anchor_num=anchor_num)
    esitmation_embedding_projection.save(f"{save_dir}/EstimationEmbeddingProjection_{anchor_num}anchors_seed{seed}.pt")

    print("\nTest:")
    with torch.no_grad():
        test_loss = esitmation_embedding_projection.evaluate(test_dataset[:][0], test_dataset[:][1]).item()
        print(f"Test on Estimation Embedding Projection: {test_loss:.4f}")

        estimate_pred_embedding = esitmation_embedding_projection.transform(test_dataset[:][0])        
        col = 1000
        print("Hisotram Comparision for Estimation:")
        compare_histogram(estimate_pred_embedding[:,col], test_dataset[:][1][:,col], 10)

        print("Matrix Scale Ratio for Estimation:")
        print(compute_matrix_scale(estimate_pred_embedding) / compute_matrix_scale(test_dataset[:][1]))

        print(f"\nMean:")
        print(f"Target: {test_dataset[:][1].mean(dim=0)}")
        print(f"Estimation: {estimate_pred_embedding.mean(dim=0)}")

        print(f"\nVar:")
        print(f"Target: {test_dataset[:][1].var(dim=0)}")
        print(f"Estimation: {estimate_pred_embedding.var(dim=0)}")