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

from utils.EmbeddingProjection.train_v2 import DeepEmbeddingProjection, EstimationEmbeddingProjection


class EmbeddingProjectionFuser():
    """
    DeepFuser fuses the hidden states of multiple models in the relative space 
        and returns the absolute representation of the fusion result.
    """
    def __init__(self,  
                 model_list,
                 layer_alignment,
                 embedding_projection_path = "",
                 device_compute="cuda:0",
                 ensembel_weights=None,
                 subspace_ratio=1):
        """
        anchor_embeddings_list: with shape (layer_num + 1, anchor_num, dimension)
        """
        model_num = len(model_list)

        self.model_list = model_list
        self.device_compute = device_compute

        if ensembel_weights is None:
            ensembel_weights = [1 / model_num] * model_num
        self.ensembel_weights = torch.tensor(ensembel_weights).to(device_compute)
        self.main_hidden_dim = self.model_list[0].embed_tokens.embedding_dim
        self.neuron_level_ensemble_weights = self.ensembel_weights.unsqueeze(dim=-1).repeat([1, self.main_hidden_dim])

        # 1. Load the Embedding Projection
        # embedding_projection = DeepEmbeddingProjection(4096, 5120, 2048)
        embedding_projection = EstimationEmbeddingProjection()
        # embedding_projection.load("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection/DeepEmbeddingProjection.pt")
        # embedding_projection.load("/data7/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection/DeepEmbeddingProjection.pt")

        embedding_projection.load(embedding_projection_path, device="cuda:0")
        # embedding_projection.load("/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/embedding_projection/EstimationEmbeddingProjection.pt", device="cuda:0")
        self.embedding_projection = embedding_projection

        # Modify neuron-level ensemble weights
        pdb.set_trace()
        selected_neuron_indices = torch.zeros_like(self.neuron_level_ensemble_weights)
        selected_neuron_indices[0] = 1  # Use all the neurons of the main model
        alignment_scores = embedding_projection.get_alignment_scores()
        assert len(alignment_scores) == main_hidden_dim
        alignment_scores.topk(int(main_hidden_dim * subspace_ratio))[1]
        print(f"Size of Subspace: {int(main_hidden_dim * subspace_ratio)}")
        selected_neuron_indices[1][selected_neuron_indices] = 1
        self.neuron_level_ensemble_weights = self.neuron_level_ensemble_weights * selected_neuron_indices
        self.neuron_level_ensemble_weights /= self.neuron_level_ensemble_weights.sum(dim=0)
        
        # embedding_projection.get_alignment_scores())

        # 3. Evaluate the optimal transformation matrix
        # print("Misalignmen of Baseline:")
        # print(((self.anchor_embeddings_list[0][torch.randperm(len(self.anchor_embeddings_list[0]))] - self.anchor_embeddings_list[0]).abs()).mean())
        # print("Optimal Misalignment of Optimal Transformation")

        # aligned_embedding = self.transform_to_main_space(self.anchor_embeddings_list[1], 1)
        # print(((self.anchor_embeddings_list[0] - aligned_embedding).abs()).mean())

        # col = 1000
        # compare_histogram(self.anchor_embeddings_list[0][:,col], aligned_embedding[:,col], 15)
        # misalignment = (aligned_embedding - self.anchor_embeddings_list[0]).abs().mean()
        # print(f"Mean Misglignment: {misalignment}")
        # print("Msialignment Distribution:")
        # print_histogram(misalignment, 10)
        
    def transform_to_main_space(self, src_embeddings, model_id):

        src_embeddings = src_embeddings.to(torch.float32)
        with torch.no_grad():
            predicted_embedding = self.embedding_projection.transform(src_embeddings)

        return predicted_embedding

    def weighted_sum(self, embed_list):
        """
        relative_embed_list: (N, T, A)

        return: (T, A)
        """
        embed_list = torch.stack(embed_list)
        # embed_list = embed_list * self.ensembel_weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
        pdb.set_trace()
        embed_list = embed_list * neuron_level_ensemble_weights
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
