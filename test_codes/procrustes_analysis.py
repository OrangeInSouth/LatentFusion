import numpy as np
import pdb

def procrustes_analysis(tgt_embeddings, src_embeddings):
    """
    Perform Procrustes Analysis to find the best mapping from src_embeddings to tgt_embeddings.

    Parameters:
    tgt_embeddings: numpy.ndarray of shape (N, d1)
        The target embedding matrix.
    src_embeddings: numpy.ndarray of shape (N, d2)
        The source embedding matrix.

    Returns:
    aligned_src_embeddings: numpy.ndarray of shape (N, d1)
        The source embeddings after applying the optimal rotation and scaling.
    R: numpy.ndarray of shape (d2, d1)
        The optimal rotation matrix.
    s: float
        The optimal scaling factor.
    t: numpy.ndarray of shape (d1,)
        The translation vector.
    """

    # Center the embeddings (subtract mean)
    tgt_mean = np.mean(tgt_embeddings, axis=0)
    src_mean = np.mean(src_embeddings, axis=0)
    
    tgt_centered = tgt_embeddings - tgt_mean
    src_centered = src_embeddings - src_mean

    # Singular Value Decomposition (SVD) to find the optimal rotation
    U, _, Vt = np.linalg.svd(np.dot(src_centered.T, tgt_centered))
    
    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Compute scale
    scale_num = np.trace(np.dot(np.dot(src_centered, R), tgt_centered.T))
    scale_den = np.trace(np.dot(src_centered.T, src_centered))
    s = scale_num / scale_den

    # Apply transformation
    # aligned_src_embeddings = s * np.dot(src_centered, R)
    aligned_src_embeddings = np.dot(src_centered, R)
    pdb.set_trace()
    
    # Translate to match tgt_embeddings
    aligned_src_embeddings += tgt_mean

    # Compute translation vector
    t = tgt_mean - s * np.dot(src_mean, R)

    return aligned_src_embeddings, R, s, t

# Example usage:
tgt_embeddings = np.random.rand(10, 3)  # Example target embeddings
src_embeddings = np.random.rand(10, 3)  # Example source embeddings

aligned_src_embeddings, R, s, t = procrustes_analysis(tgt_embeddings, src_embeddings)

print("Aligned Source Embeddings:\n", aligned_src_embeddings)
print("Optimal Rotation Matrix:\n", R)
print("Optimal Scaling Factor:\n", s)
print("Translation Vector:\n", t)