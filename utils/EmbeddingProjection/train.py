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

torch.manual_seed(1)


class EmbeddingProjection(object):
    def __init__(self) -> None:
        self.src_mean = None
        self.src_std = None
        self.tgt_mean = None
        self.tgt_std = None
        self.transformation_weights = None
        self.tgt_dtype = None
    
    def transform(self, embeddings):
        embeddings = self.normalize(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.denormalization(embeddings)

        embeddings = embeddings.to(self.tgt_dtype)
        return embeddings
    
    def normalize(self, embeddings):
        if embeddings.device != self.src_mean.device:
            embeddings = embeddings.to(self.src_mean.device)
        embeddings =(embeddings - self.src_mean) / self.src_std
        return embeddings
    
    def denormalization(self, embeddings):
        if embeddings.device != self.tgt_std.device:
            embeddings = embeddings.to(self.tgt_std.device)
        embeddings = embeddings * self.tgt_std + self.tgt_mean
        return embeddings
    
    def fit(self, src_embeddings, tgt_embeddings):
        raise NotImplementedError
    
    def projection(self, embeddings):
        raise NotImplementedError
    
    def evaluate(self, src_embeddings, tgt_embeddings):
        src_embeddings = src_embeddings.to(torch.float32)
        tgt_embeddings = tgt_embeddings.to(torch.float32)
        with torch.no_grad():
            predicted_embedding = self.transform(src_embeddings)
            
            loss = (tgt_embeddings - predicted_embedding).abs().mean()
            return loss

    def static_states(self):
        states = {
            "src_mean":self.src_mean,
            "tgt_mean":self.tgt_mean,
            "src_std":self.src_std,
            "tgt_std":self.tgt_std,
            "transformation_weights":self.transformation_weights,
        }
        return states

    def load(self, checkpoint_path, device=None):
        if device is None:
            state = torch.load(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location=device)

        self.src_mean = state.pop("src_mean")
        self.tgt_mean = state.pop("tgt_mean")
        self.src_std = state.pop("src_std")
        self.tgt_std = state.pop("tgt_std")
        self.transformation_weights = state.pop("transformation_weights")

        return state
    

class DeepEmbeddingProjection(EmbeddingProjection):

    def __init__(self, inner_dim=16384):
        super().__init__()
        self.inner_dim = inner_dim

    def init_M(self, M):
        self.M = M

    def fit(self, training_set, val_set, max_epochs=100, batch_size=100, lr=0.01):
        """
        src_embeddings: torch tensor with shape (M, d1)
        tgt_embeddings: torch tensor with shape (M, d2)

        Outputs:
        src_mean: Mean of source embeddings across dimensions
        src_std: Std deviation of source embeddings across dimensions
        tgt_mean: Mean of target embeddings across dimensions
        tgt_std: Std deviation of target embeddings across dimensions
        M: Optimal transformation matrix from source to target space
        b: Optimal bias for the transformation
        """
        start_time = time.time()

        self.tgt_dtype = training_set[:][1].dtype
        # device = training_set[:][0].device
        device = "cuda:0"

        src_embeddings = training_set[:][0].to(torch.float32)
        tgt_embeddings = training_set[:][1].to(torch.float32)
        train_dataset = TensorDataset(src_embeddings, tgt_embeddings)

        # 1. Split dataset into train, validation, and test sets (8:1:1) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 2. Compute means and standard deviations for normalization
        src_mean = src_embeddings.mean(dim=0, keepdim=True)
        src_std = src_embeddings.std(dim=0, keepdim=True)
        tgt_mean = tgt_embeddings.mean(dim=0, keepdim=True)
        tgt_std = tgt_embeddings.std(dim=0, keepdim=True)
        
        # Standardize the embeddings
        src_embeddings = (src_embeddings - src_mean) / src_std
        tgt_embeddings = (tgt_embeddings - tgt_mean) / tgt_std

        # 3. Initialize M and b, optimizer, and learning rate scheduler
        d1 = src_embeddings.shape[1]
        d2 = tgt_embeddings.shape[1]

        
        # if self.M is None:
        #     M = nn.Parameter(torch.randn(d1, d2, device=device), requires_grad=True)
        # else:
        #     M = nn.Parameter(self.M.to(device), requires_grad=True)
        M_in = nn.Parameter(torch.randn(d1, self.inner_dim, device=device), requires_grad=True)
        M_out = nn.Parameter(torch.randn(self.inner_dim, d2, device=device), requires_grad=True)
        b = nn.Parameter(torch.zeros(d2, device=device), requires_grad=True)
        self.transformation_weights = {"M_in": M_in,
                                        "M_out": M_out,
                                        "b": b}
        
        optimizer = optim.Adam(list(self.transformation_weights.values()), lr=lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs*2)

        best_val_loss = float('inf')
        best_state = None

        # 4. Training loop
        for epoch in tqdm(range(max_epochs)):
            train_loss = 0.0
            for src_batch, tgt_batch in train_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                optimizer.zero_grad()
                
                # Compute predictions
                preds = self.projection(src_batch)
                # preds = torch.matmul(src_batch, M) + b
                
                # Compute loss (MSE)
                mse_loss = nn.functional.mse_loss(preds, tgt_batch)
                exp_loss = nn.functional.mse_loss(preds.mean(dim=0), tgt_batch.mean(dim=0))
                val_loss = nn.functional.mse_loss(preds.std(dim=0), tgt_batch.std(dim=0))
                loss = mse_loss + exp_loss + val_loss

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.src_mean = src_mean.to(device)
            self.src_std = src_std.to(device)
            self.tgt_mean = tgt_mean.to(device)
            self.tgt_std = tgt_std.to(device)

            # Validation
            with torch.no_grad():
                val_loss = self.evaluate(val_dataset[:][0].to(device), val_dataset[:][1].to(device)).item()
            scheduler.step(val_loss)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'M_in': M_in.detach().clone(),
                    'M_out': M_out.detach().clone(),
                    'b': b.detach().clone(),
                }

            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 6. Load the best model and evaluate on the test set
        # M_in = best_state['M']
        # b = best_state['b']

        self.src_mean = src_mean.to(device)
        self.src_std = src_std.to(device)
        self.tgt_mean = tgt_mean.to(device)
        self.tgt_std = tgt_std.to(device)
        self.transformation_weights = best_state

        # # Post Rescale:
        # pred_embed = self.projection(src_embeddings)
        # rescale_ratio = compute_matrix_scale(pred_embed[:10000]) / compute_matrix_scale(tgt_embeddings[:10000])
        # self.transformation_weights["M_out"] = self.transformation_weights["M_out"] / rescale_ratio

        end_time = time.time()
        consuming_time = end_time - start_time
        print(f"Training Time: {(consuming_time)//3600}h:{consuming_time%3600}s")

    def projection(self, embeddings):
        M_in = self.transformation_weights['M_in']
        M_out = self.transformation_weights['M_out']
        b = self.transformation_weights['b']
        if embeddings.device != M_in.device:
            embeddings = embeddings.to(M_in.device)

        transformed_embeddings = embeddings @ M_in / M_in.shape[1]
        transformed_embeddings = (transformed_embeddings - transformed_embeddings.mean(dim=-1).unsqueeze(dim=-1)) / transformed_embeddings.std(dim=-1).unsqueeze(dim=-1)
        transformed_embeddings = transformed_embeddings @ M_out / M_out.shape[1]
        return transformed_embeddings + b

    def save(self, checkpoint_name):
        state = self.static_states()
        torch.save(state, checkpoint_name)


class EstimationEmbeddingProjection(EmbeddingProjection):
    def fit(self, training_set, anchor_num=10000):
        """
        tgt_embeddings: (N, d1)
        src_embeddings: (N, d2)

        return optimal_transformation (d2, d1)
        """
        start_time = time.time()

        self.tgt_dtype = training_set[:][1].dtype
        device = "cpu"
        
        # Anchor Selection
        random.seed(1)
        rand_indices = list(range(len(train_dataset)))
        random.shuffle(rand_indices)
        selected_anchor_indices = rand_indices[:anchor_num]

        src_embeddings = train_dataset[selected_anchor_indices][0].to(torch.float32)
        tgt_embeddings = train_dataset[selected_anchor_indices][1].to(torch.float32)
        
        # z-score
        src_mean = src_embeddings.mean(dim=0, keepdim=True)
        src_std = src_embeddings.std(dim=0, keepdim=True)
        tgt_mean = tgt_embeddings.mean(dim=0, keepdim=True)
        tgt_std = tgt_embeddings.std(dim=0, keepdim=True)

        Y = (tgt_embeddings - tgt_mean) / tgt_std
        X = (src_embeddings - src_mean) / src_std

        # to numpy
        Y = Y.cpu().numpy()
        X = X.cpu().numpy()

        # zero padding
        d_src = X.shape[-1]
        d_tgt = Y.shape[-1]
        max_dim = max(d_src, d_tgt)
        Y = np.pad(Y, ((0, 0), (0, max_dim - d_tgt)), mode='constant')
        X = np.pad(X, ((0, 0), (0, max_dim - d_src)), mode='constant')


        # Calculate optimal transformation
        Y0T = Y.T
        X0T = X.T
        U, s, Vt = np.linalg.svd(np.dot(Y0T, X0T.T))
        # Calculate the rotation matrix
        # R = np.dot(Vt.T, U.T)
        R = np.dot(U, Vt)
        optimal_transformation = R.T

        # Calculate the rescale ratio
        target_scale = compute_matrix_scale(Y)
        prediction_scale = compute_matrix_scale(X@optimal_transformation)
        rescale_ratio = target_scale / prediction_scale
        optimal_transformation *= rescale_ratio

        # de-padding
        optimal_transformation = optimal_transformation[:d_src, :d_tgt]
        
        optimal_transformation = torch.from_numpy(optimal_transformation).to(src_embeddings.device)

        self.src_mean = src_mean
        self.src_std = src_std
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        self.transformation_weights = {"M": optimal_transformation}

        end_time = time.time()
        consuming_time = end_time - start_time
        print(f"Training Time: {(consuming_time)//3600}h:{consuming_time%3600}s")

    def projection(self, embeddings):
        M = self.transformation_weights['M']
        if embeddings.device != M.device:
            embeddings = embeddings.to(M.device)
        return embeddings @ M

    def save(self, checkpoint_name):
        state = self.static_states()
        torch.save(state, checkpoint_name)
        
        
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

    anchor_embeddings_path = "/data1/cpfu/ychuang/llama2-13b_mistral-7b_200000anchors_seed1.pt"
    save_dir = "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection"
    state = torch.load(anchor_embeddings_path)
    src_embeddings = state["mistral-7b"][32]
    tgt_embeddings = state["llama2-13b"][40]

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
    deep_embedding_projection = DeepEmbeddingProjection(inner_dim=inner_dim)
    pdb.set_trace()
    # deep_embedding_projection.init_M(esitmation_embedding_projection.transformation_weights["M"])
    deep_embedding_projection.fit(train_dataset, val_dataset, max_epochs=max_epoch, batch_size=batch_size, lr=lr)
    deep_embedding_projection.save(f"{save_dir}/DeepEmbeddingProjection.pt")

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
        compute_matrix_scale(deep_pred_embedding) / compute_matrix_scale(test_dataset[:][1])
        print("Matrix Scale Ratio for Estimation:")
        compute_matrix_scale(estimate_pred_embedding) / compute_matrix_scale(test_dataset[:][1])


    
    # print("Case Study:")
    # print(f"Target: {tgt_embeddings[0][:10]}")
    # print("Deep Embedding Projection:", deep_embedding_projection.transform(src_embeddings[0])[0,:10])
    # print("Estimation Embedding Projection:", esitmation_embedding_projection.transform(src_embeddings[0])[0,:10])
