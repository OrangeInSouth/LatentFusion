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
    parser.add_argument('--inner-dim', type=int, default=2048, help='Inner dimension size (default: 2048)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 1)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size (default: 1000)')
    parser.add_argument('--debug', action='store_true', help="debug模式")

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
    debug_mode = args.debug

    # if debug_mode:
    #     anchor_embeddings_path = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/anchor_embeddings/llama2-13b_mistral-7b_20000anchors_seed1.pt"
    # else:
    #     anchor_embeddings_path = "/data1/cpfu/ychuang/llama2-13b_mistral-7b_200000anchors_seed1.pt"
    anchor_embeddings_path = "/data1/cpfu/ychuang/llama2-13b_mistral-7b_200000anchors_seed1_layer40-32.pt"
    save_dir = "/data7/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection"
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
    # deep_embedding_projection.fit(train_dataset, val_dataset, max_epochs=max_epoch, batch_size=batch_size, lr=lr)
    # deep_embedding_projection.save(f"{save_dir}/DeepEmbeddingProjection.pt")

    print("Estimation:")
    esitmation_embedding_projection = EstimationEmbeddingProjection()
    esitmation_embedding_projection.fit(train_dataset, anchor_num=200000)
    esitmation_embedding_projection.save(f"{save_dir}/EstimationEmbeddingProjection.pt")

    print("\nTest:")
    with torch.no_grad():
        test_loss = deep_embedding_projection.evaluate(test_dataset[:][0].to("cuda:0"), test_dataset[:][1].to("cuda:0")).item()
        print(f"Test on Deep Embedding Projection: {test_loss:.4f}")
        test_loss = esitmation_embedding_projection.evaluate(test_dataset[:][0], test_dataset[:][1]).item()
        print(f"Test on Estimation Embedding Projection: {test_loss:.4f}")

        deep_pred_embedding = deep_embedding_projection.transform(test_dataset[:][0].to("cuda:0"))
        estimate_pred_embedding = esitmation_embedding_projection.transform(test_dataset[:][0])

        print("Hisotram Comparision for Deep:")
        col = 1000
        compare_histogram(deep_pred_embedding[:,col], test_dataset[:][1][:,col], 10)
        print("Hisotram Comparision for Estimation:")
        compare_histogram(estimate_pred_embedding[:,col], test_dataset[:][1][:,col], 10)

        print("Matrix Scale Ratio for Deep:")
        print(compute_matrix_scale(deep_pred_embedding) / compute_matrix_scale(test_dataset[:][1]))
        print("Matrix Scale Ratio for Estimation:")
        print(compute_matrix_scale(estimate_pred_embedding) / compute_matrix_scale(test_dataset[:][1]))

        print(f"\nMean:")
        print(f"Target: {test_dataset[:][1].mean(dim=0)}")
        print(f"Estimation: {estimate_pred_embedding.mean(dim=0)}")
        print(f"Deep: {deep_pred_embedding.mean(dim=0)}")

        print(f"\nVar:")
        print(f"Target: {test_dataset[:][1].var(dim=0)}")
        print(f"Estimation: {estimate_pred_embedding.var(dim=0)}")
        print(f"Deep: {deep_pred_embedding.var(dim=0)}")

        pdb.set_trace()


    
    # print("Case Study:")
    # print(f"Target: {tgt_embeddings[0][:10]}")
    # print("Deep Embedding Projection:", deep_embedding_projection.transform(src_embeddings[0])[0,:10])
    # print("Estimation Embedding Projection:", esitmation_embedding_projection.transform(src_embeddings[0])[0,:10])
