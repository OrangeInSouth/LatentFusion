import sys
import argparse
sys.path.append("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang")
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
torch.manual_seed(1)

        
def parse_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")

    # Add arguments with default values
    parser.add_argument('--max-epoch', type=int, default=100, help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--inner-dim', type=int, default=16384, help='Inner dimension size (default: 2048)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 1)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size (default: 1000)')

    # Parse the arguments
    args = parser.parse_args()
    return args

# Example usage
# anchor_embeddings_path = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/llama2-13b_mistral-7b_20000anchors_seed1.pt"

if __name__ == "__main__":
    args = parse_args()
    lr = args.lr
    batch_size = args.batch_size
    inner_dim = args.inner_dim
    max_epoch = args.max_epoch

    anchor_embeddings_path = "/data1/cpfu/ychuang/llama2-13b_mistral-7b_200000anchors_seed1_layer40-32.pt"
    embedding_projection_dir = "/data7/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection"
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
    # train_dataset_size = 10000
    # train_dataset = train_dataset[:train_dataset_size]

    # 2. training
    print("Deep Emebdding Projection:")
    src_dim = src_embeddings.shape[-1]
    tgt_dim = tgt_embeddings.shape[-1]
    deep_embedding_projection = DeepEmbeddingProjection(src_dim, tgt_dim, inner_dim=inner_dim)
    deep_embedding_projection.load(f"{embedding_projection_dir}/DeepEmbeddingProjection.pt")

    print("Estimation:")
    esitmation_embedding_projection = EstimationEmbeddingProjection()
    esitmation_embedding_projection.load(f"{embedding_projection_dir}/EstimationEmbeddingProjection.pt")


    print("\nTest:")
    with torch.no_grad():
        test_loss = deep_embedding_projection.evaluate(test_dataset[:][0].to("cuda:0"), test_dataset[:][1].to("cuda:0")).item()
        print(f"Test on Deep Embedding Projection: {test_loss:.4f}")
        test_loss = esitmation_embedding_projection.evaluate(test_dataset[:][0], test_dataset[:][1]).item()
        print(f"Test on Estimation Embedding Projection: {test_loss:.4f}")

        deep_pred_embedding = deep_embedding_projection.transform(test_dataset[:][0].to("cuda:0"))
        estimate_pred_embedding = esitmation_embedding_projection.transform(test_dataset[:][0])

        print("Hisotram Comparision:")
        print("Deep:")
        col = 1000
        compare_histogram(test_dataset[:][1][:,col], deep_pred_embedding[:,col], 10)
        print("Estimation:")
        compare_histogram(test_dataset[:][1][:,col], estimate_pred_embedding[:,col], 10)

        print("\nMatrix Scale Ratio:")
        print("Deep:")
        print(compute_matrix_scale(deep_pred_embedding) / compute_matrix_scale(test_dataset[:][1]))
        print("Estimation:")
        print(compute_matrix_scale(estimate_pred_embedding) / compute_matrix_scale(test_dataset[:][1]))

        print("\nMisalignment of Each Neuron:")
        deep_pred_misalignment = (deep_pred_embedding - test_dataset[:][1].to("cuda:0")).mean(dim=0)
        estimate_pred_misalignment = (estimate_pred_embedding - test_dataset[:][1]).mean(dim=0)
        compare_histogram(deep_pred_misalignment, estimate_pred_misalignment, 20)



    
    # print("Case Study:")
    # print(f"Target: {tgt_embeddings[0][:10]}")
    # print("Deep Embedding Projection:", deep_embedding_projection.transform(src_embeddings[0])[0,:10])
    # print("Estimation Embedding Projection:", esitmation_embedding_projection.transform(src_embeddings[0])[0,:10])
