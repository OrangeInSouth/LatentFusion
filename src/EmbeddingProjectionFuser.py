import pdb

import torch
from torch import nn
from torch.optim import lr_scheduler
from utils.block_operations import block_dot, block_cosine_similarity
from utils import avg
import random
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from utils.distribution_utils import print_histogram, compare_histogram


class EmbeddingProjectionFuser():
    """
    DeepFuser fuses the hidden states of multiple models in the relative space 
        and returns the absolute representation of the fusion result.
    """
    def __init__(self,  
                 model_list,
                 layer_alignment,
                 anchor_embeddings_list,
                 anchor_num,
                 device_compute="cuda:0",
                 ensembel_weights=None):
        """
        anchor_embeddings_list: with shape (layer_num + 1, anchor_num, dimension)
        """
        model_num = len(model_list)

        anchor_embeddings_list = [anchor_embeddings_list[i][layer_alignment[i]].to(device_compute) for i in range(model_num)]
        self.anchor_embeddings_list = anchor_embeddings_list

        self.model_list = model_list
        self.device_compute = device_compute
        self.anchor_num = anchor_num

        if ensembel_weights is None:
            ensembel_weights = [1 / model_num] * model_num
        self.ensembel_weights = torch.tensor(ensembel_weights).to(device_compute)

        # 1. Compute the expectation and the standard derivation
        self.mean_list = [m.mean(dim=0) for m in anchor_embeddings_list]
        self.std_list = [m.std(dim=0) for m in anchor_embeddings_list]

        # 2. Compute the optimal transformation matrix
        self.optimal_transformation_list = []
        for i in range(1, model_num):
            self.optimal_transformation_list.append(self.get_optimal_transformation(anchor_embeddings_list[0], anchor_embeddings_list[i], i))

        # 3. Evaluate the optimal transformation matrix
        print("Misalignmen of Baseline:")
        print(((self.anchor_embeddings_list[0][torch.randperm(len(self.anchor_embeddings_list[0]))] - self.anchor_embeddings_list[0]).abs()).mean())
        print("Optimal Misalignment of Optimal Transformation")
        aligned_embedding = self.transform_to_main_space(self.anchor_embeddings_list[1], 1)
        print(((self.anchor_embeddings_list[0] - aligned_embedding).abs()).mean())

        pdb.set_trace()
        col = 1000
        compare_histogram(self.anchor_embeddings_list[0][:,col], aligned_embedding[:,col], 15)
        misalignment = (aligned_embedding - self.anchor_embeddings_list[0]).abs().mean()
        print(f"Mean Misglignment: {misalignment}")
        # print("Msialignment Distribution:")
        # pdb.set_trace()
        # print_histogram(misalignment, 10)
        


    def get_optimal_transformation(self, tgt_embeddings, src_embeddings, src_model_id):
        """
        tgt_embeddings: (N, d1)
        src_embeddings: (N, d2)

        return optimal_transformation (d2, d1)
        """

        assert len(tgt_embeddings) == len(src_embeddings)

        # Anchor Selection
        random.seed(1)
        rand_indices = list(range(len(tgt_embeddings)))
        random.shuffle(rand_indices)
        selected_anchor_indices = rand_indices[:self.anchor_num]
        tgt_embeddings = tgt_embeddings[selected_anchor_indices]
        src_embeddings = src_embeddings[selected_anchor_indices]

        # z-score
        tgt_mean = self.mean_list[0]
        tgt_std = self.std_list[0]
        src_mean = self.mean_list[src_model_id]
        src_std = self.std_list[src_model_id]

        Y = (tgt_embeddings - tgt_mean) / tgt_std
        X = (src_embeddings - src_mean) / src_std

        # to numpy
        Y = Y.cpu().to(torch.float32).numpy()
        X = X.cpu().to(torch.float32).numpy()

        # zero padding
        d_src = X.shape[-1]
        d_tgt = Y.shape[-1]
        max_dim = max(d_src, d_tgt)
        Y = np.pad(Y, ((0, 0), (0, max_dim - d_tgt)), mode='constant')
        X = np.pad(X, ((0, 0), (0, max_dim - d_src)), mode='constant')


        # Calculate optimal transformation
        # pdb.set_trace()
        # optimal_transformation = compute_optimal_transformation(Y, X)
        Y0T = Y.T
        X0T = X.T
        U, s, Vt = np.linalg.svd(np.dot(Y0T, X0T.T))
        # Calculate the rotation matrix
        # R = np.dot(Vt.T, U.T)
        R = np.dot(U, Vt)
        optimal_transformation = R.T
        # pdb.set_trace()
        # orthogonal_procrustes()

        # Calculate the rescale ratio
        target_scale = compute_matrix_scale(Y)
        prediction_scale = compute_matrix_scale(X@optimal_transformation)
        rescale_ratio = target_scale / prediction_scale
        optimal_transformation = rescale_ratio * optimal_transformation

        # reshape back
        optimal_transformation = optimal_transformation[:d_src, :d_tgt]

        # pdb.set_trace()
        
        optimal_transformation = torch.from_numpy(optimal_transformation).to(self.device_compute)
        
        return optimal_transformation

    def transform_to_main_space(self, embeddings, model_id):
        # 1. scaling (z-score)
        embeddings = (embeddings - self.mean_list[model_id]) / self.std_list[model_id]

        # 2. transform
        transformation_matrix = self.optimal_transformation_list[model_id - 1]
        transformed_embeddings = embeddings.to(transformation_matrix.dtype) @ transformation_matrix
        # pdb.set_trace()

        # 3. de-scaling
        transformed_embeddings = transformed_embeddings * self.std_list[0] + self.mean_list[0]

        return transformed_embeddings

    def weighted_sum(self, embed_list):
        """
        relative_embed_list: (N, T, A)

        return: (T, A)
        """
        embed_list = torch.stack(embed_list)
        embed_list = embed_list * self.ensembel_weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
        aggregated_relative_embed = embed_list.sum(dim=0)
        return aggregated_relative_embed

    def fuse(self, hidden_state_list):
        # 1. transform the representations into the space of the main model
        aligned_embed_list = [hidden_state_list[0]]
        for i in range(1, len(hidden_state_list)):
            aligned_embed_list.append(self.transform_to_main_space(hidden_state_list[i], i))
        # 2. aggregation
        aggregation_res = self.weighted_sum(aligned_embed_list)

        # 3. data type
        aggregation_res = aggregation_res.to(hidden_state_list[0].dtype)

        return aggregation_res


def compute_matrix_scale(m):
    if isinstance(m, torch.Tensor):
        m = m.to(torch.float32)
    return (m**2).sum()**0.5 / m.shape[0]
