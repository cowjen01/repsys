from jax import numpy as jnp


def recall(k_mask, sort_indices, x_true_bin):
    k = k_mask.sum()
    # an array of row number indices
    row_indices = jnp.arange(x_true_bin.shape[0])[:, jnp.newaxis]
    # increase col indices by 1 to make zero index available as a null value
    sort_indices = sort_indices + 1
    # keep only first K cols for each row, the rest set to zero
    col_indices = jnp.where(k_mask, sort_indices, 0)
    # create a binary matrix from indices
    # the matrix has an increased dimension by 1 to hold the increased indices
    predict_matrix_shape = (x_true_bin.shape[0], x_true_bin.shape[1] + 1)
    predict_matrix = jnp.zeros(predict_matrix_shape, dtype=bool)
    predict_matrix = predict_matrix.at[row_indices, col_indices].set(True)
    # remove the first column that holds unused indices
    predict_matrix = predict_matrix[:, 1:]
    tmp = (jnp.logical_and(x_true_bin, predict_matrix).sum(axis=1)).astype(jnp.float32)
    # divide a sum of the agreed indices by a sum of the true indices
    # to avoid dividing by zero, use the K value as a minimum
    return tmp / jnp.minimum(k, x_true_bin.sum(axis=1))


def ndcg(ndcg_mask, sort_indices, x_true_clipped):
    K = sort_indices.shape[1]
    rows_idx = jnp.arange(x_true_clipped.shape[0])[:, jnp.newaxis]
    tp = 1.0 / jnp.log2(jnp.arange(2, K + 2))
    # the slowest part ...
    dcg = (x_true_clipped[rows_idx, sort_indices] * tp).sum(axis=1)
    # null all positions, where mask is False (keep the rest)
    idcg = jnp.where(ndcg_mask, tp, 0).sum(axis=1)
    return dcg / idcg
