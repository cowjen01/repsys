from jax import jit
import jax.numpy as jnp


@jit
def recall_jax(X_pred_idx, X_true_binary, row_idx):
    X_pred_binary = jnp.zeros_like(X_true_binary, dtype=bool)
    X_pred_binary = X_pred_binary.at[row_idx, X_pred_idx].set(True)
    tmp = (jnp.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        jnp.float32
    )
    recall = tmp / jnp.minimum(X_pred_idx.shape[1], X_true_binary.sum(axis=1))
    return recall
