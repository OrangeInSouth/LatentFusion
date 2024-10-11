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
        if isinstance(self.transformation_weights, nn.Module):
            self.transformation_weights.eval()
        src_embeddings = src_embeddings.to(torch.float32)
        tgt_embeddings = tgt_embeddings.to(torch.float32)
        with torch.no_grad():
            predicted_embedding = self.transform(src_embeddings)
            
            # loss = (tgt_embeddings - predicted_embedding).abs().mean()
            loss = torch.nn.functional.mse_loss(predicted_embedding, tgt_embeddings)
            return loss
    

class DeepEmbeddingProjection(EmbeddingProjection):

    def __init__(self, src_dim, tgt_dim, inner_dim=16384, device="cuda:0"):
        super().__init__()
        self.inner_dim = inner_dim
        self.transformation_weights = EmbeddingProjectionNetwork(src_dim, tgt_dim, self.inner_dim, device=device)
        self.transformation_weights = EmbeddingLinearProjectionNetwork(src_dim, tgt_dim, self.inner_dim, device=device)

    # def init_M(self, M):
    #     self.M = M

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

        # 2. Preprocess the training data
        # 2.1 Compute means and standard deviations
        src_mean = src_embeddings.mean(dim=0, keepdim=True)
        src_std = src_embeddings.std(dim=0, keepdim=True)
        tgt_mean = tgt_embeddings.mean(dim=0, keepdim=True)
        tgt_std = tgt_embeddings.std(dim=0, keepdim=True)
        self.src_mean = src_mean.to(device)
        self.src_std = src_std.to(device)
        self.tgt_mean = tgt_mean.to(device)
        self.tgt_std = tgt_std.to(device)
        
        # 2.2 Standardize the embeddings
        src_embeddings = (src_embeddings - src_mean) / src_std
        # tgt_embeddings = (tgt_embeddings - tgt_mean) / tgt_std

        # 3. Initialize the projection network, optimizer, and learning rate scheduler        
        optimizer = optim.Adam(self.transformation_weights.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs*2)

        best_val_loss = float('inf')
        best_epoch = -1
        best_checkpoint = None

        # 4. Training loop
        for epoch in tqdm(range(max_epochs)):
            self.transformation_weights.train()
            train_loss = 0.0
            for src_batch, tgt_batch in train_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                optimizer.zero_grad()
                
                # Compute predictions
                preds = self.projection(src_batch)
                
                # Compute loss (MSE)
                mse_loss = nn.functional.mse_loss(preds, tgt_batch)
                # exp_loss = nn.functional.mse_loss(preds.mean(dim=0), tgt_batch.mean(dim=0))
                # val_loss = nn.functional.mse_loss(preds.std(dim=0), tgt_batch.std(dim=0))
                loss = mse_loss #+ exp_loss + val_loss

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.transformation_weights.eval()
            with torch.no_grad():
                val_loss = self.evaluate(val_dataset[:][0].to(device), val_dataset[:][1].to(device)).item()
            scheduler.step(val_loss)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_checkpoint = {key: value.detach().clone() for key, value in self.transformation_weights.state_dict().items()}

            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 5. Load the best model and evaluate on the test set
        print(f"best epoch: {best_epoch}")
        print(f"best valid loss: {best_val_loss}")
        self.transformation_weights.load_state_dict(best_checkpoint)

        pdb.set_trace()

        # Post processing:
        post_processing = True
        if post_processing:
            sampled_src_embedding = src_embeddings[:20000].to(device)
            sampled_tgt_embedding = tgt_embeddings[:20000].to(device)

            pred_embed = self.projection(sampled_src_embedding)


            # 分布对齐的重缩放、平移
            with torch.no_grad():
                std_ratio = sampled_tgt_embedding.std(dim=0) / pred_embed.std(dim=0)
                self.transformation_weights.scale_ratio.copy_(std_ratio)      

                pred_embed = self.projection(sampled_src_embedding)
                mean_distance = sampled_tgt_embedding.mean(dim=0) - pred_embed.mean(dim=0)
                self.transformation_weights.center_bias.copy_(mean_distance)

            # embedding尺度对齐的重缩放
            # rescale_ratio = compute_matrix_scale(pred_embed) / compute_matrix_scale(tgt_embeddings[:10000])
            # with torch.no_grad():
            #     self.transformation_weights.M_out.copy_(self.transformation_weights.M_out / rescale_ratio)
            #     self.transformation_weights.b.copy_(self.transformation_weights.b / rescale_ratio)

        end_time = time.time()
        consuming_time = end_time - start_time
        print(f"Training Time: {(consuming_time)//3600}h:{consuming_time%3600}s")

    def transform(self, embeddings):
        embeddings = self.normalize(embeddings)
        embeddings = self.projection(embeddings)

        embeddings = embeddings.to(self.tgt_dtype)
        return embeddings
        
    def projection(self, embeddings):
        return self.transformation_weights(embeddings)
        # M_in = self.transformation_weights['M_in']
        # M_out = self.transformation_weights['M_out']
        # b = self.transformation_weights['b']
        # if embeddings.device != M_in.device:
        #     embeddings = embeddings.to(M_in.device)

        # transformed_embeddings = embeddings @ M_in / M_in.shape[1]
        # transformed_embeddings = (transformed_embeddings - transformed_embeddings.mean(dim=-1).unsqueeze(dim=-1)) / transformed_embeddings.std(dim=-1).unsqueeze(dim=-1)
        # transformed_embeddings = transformed_embeddings @ M_out / M_out.shape[1]
        # return transformed_embeddings + b

    def save(self, checkpoint_name):
        states = {
            "src_mean":self.src_mean,
            "tgt_mean":self.tgt_mean,
            "src_std":self.src_std,
            "tgt_std":self.tgt_std,
            "transformation_weights": self.transformation_weights.state_dict()
        }
        torch.save(states, checkpoint_name)

    def load(self, checkpoint_path, device=None):
        if device is None:
            state = torch.load(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location=device)

        self.src_mean = state.pop("src_mean")
        self.tgt_mean = state.pop("tgt_mean")
        self.src_std = state.pop("src_std")
        self.tgt_std = state.pop("tgt_std")
        self.transformation_weights.load_state_dict(state.pop("transformation_weights"))
        return state

class EmbeddingProjectionNetwork(nn.Module):

    def __init__(self, src_dim, tgt_dim, inner_dim, device="cuda:0"):
        super(EmbeddingProjectionNetwork, self).__init__()
        self.device = device
        self.M_in = nn.Parameter(torch.randn(src_dim, inner_dim, device=device), requires_grad=True)
        self.M_out = nn.Parameter(torch.randn(inner_dim, tgt_dim, device=device), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(inner_dim, device=device), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(tgt_dim, device=device), requires_grad=True)
        self.layernorm = nn.BatchNorm1d(inner_dim, device=device)
        self.dropout = nn.Dropout(p=0.3)
        # self.layernorm = nn.LayerNorm(inner_dim, device=device)
        # self.layernorm2 = nn.BatchNorm1d(tgt_dim, device=device)

    def forward(self, embeddings):
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)

        # M_in
        transformed_embeddings = embeddings @ self.M_in / self.M_in.shape[1]
        transformed_embeddings = transformed_embeddings + self.b1

        # layer normalize
        transformed_embeddings = self.layernorm(transformed_embeddings)
        transformed_embeddings = self.dropout(transformed_embeddings)
        # transformed_embeddings = (transformed_embeddings - transformed_embeddings.mean(dim=-1).unsqueeze(dim=-1)) / transformed_embeddings.std(dim=-1).unsqueeze(dim=-1)

        # M_out
        transformed_embeddings = transformed_embeddings @ self.M_out / self.M_out.shape[1]
        transformed_embeddings = transformed_embeddings + self.b2
        
        # transformed_embeddings = self.layernorm2(transformed_embeddings)
        
        return transformed_embeddings

class EmbeddingLinearProjectionNetwork(nn.Module):

    def __init__(self, src_dim, tgt_dim, inner_dim, device="cuda:0"):
        super(EmbeddingLinearProjectionNetwork, self).__init__()
        self.device = device
        self.M = nn.Parameter(torch.randn(src_dim, tgt_dim, device=device), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(tgt_dim, device=device), requires_grad=True)
        self.scale_ratio = nn.Parameter(torch.ones(tgt_dim, device=device), requires_grad=False)
        self.center_bias = nn.Parameter(torch.zeros(tgt_dim, device=device), requires_grad=False)
        # self.layernorm = nn.BatchNorm1d(tgt_dim, device=device)
        

    def forward(self, embeddings):
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)

        # M_in
        x = embeddings @ self.M / self.M.shape[1]
        x = x + self.b
        # transformed_embeddings = self.layernorm(transformed_embeddings)

        x = x * self.scale_ratio
        x = x + self.center_bias

        return x

    

    

    


class EstimationEmbeddingProjection(EmbeddingProjection):
    def fit(self, train_dataset, anchor_num=10000):
        """
        tgt_embeddings: (N, d1)
        src_embeddings: (N, d2)

        return optimal_transformation (d2, d1)
        """
        start_time = time.time()

        self.tgt_dtype = train_dataset[:][1].dtype
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

    def transform(self, embeddings):
        embeddings = self.normalize(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.denormalization(embeddings)

        embeddings = embeddings.to(self.tgt_dtype)
        return embeddings

    def projection(self, embeddings):
        M = self.transformation_weights['M']
        if embeddings.device != M.device:
            embeddings = embeddings.to(M.device)
        return embeddings @ M

    def save(self, checkpoint_name):
        states = {
            "src_mean":self.src_mean,
            "tgt_mean":self.tgt_mean,
            "src_std":self.src_std,
            "tgt_std":self.tgt_std,
            "transformation_weights": self.transformation_weights
        }
        torch.save(states, checkpoint_name)
        print(f"Embedding Projection is saved: {checkpoint_name}")

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