import functools
from typing import Optional

import numpy as np
import pymde
from scipy.sparse import csr_matrix

from repsys.dataset import Dataset
from repsys.helpers import set_seed


def enforce_updated(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_updated") or not self._updated:
            raise Exception("The evaluator must be updated (call update() method).")
        return func(self, *args, **kwargs)

    return wrapper


class DatasetEvaluator:
    def __init__(self, seed: int = 1234, verbose: bool = True):
        self.dataset: Optional[Dataset] = None
        self.verbose = verbose
        self.seed = seed
        self._updated = False

    def update(self, dataset: Dataset):
        self.dataset = dataset
        self._updated = True

    def _compute_embeddings(self, matrix: csr_matrix, max_samples: int, **kwargs):
        set_seed(self.seed)
        index_perm = np.random.permutation(matrix.shape[0])
        index_perm = index_perm[: max_samples]
        matrix = matrix[index_perm]

        pymde.seed(self.seed)
        mde = pymde.preserve_neighbors(matrix, init='random', verbose=self.verbose, **kwargs)
        embeddings = mde.embed(verbose=self.verbose, max_iter=1000, memory_size=50)
        embeddings = embeddings.cpu().numpy()

        return embeddings

    @enforce_updated
    def get_item_embeddings(self, split: str = 'train', max_samples: int = 5000, **kwargs):
        matrix = self.dataset.splits.get(split).complete_matrix.T
        embeddings = self._compute_embeddings(matrix, max_samples, **kwargs)

        return embeddings

    @enforce_updated
    def get_user_embeddings(self, split: str = 'train', max_samples: int = 5000, **kwargs):
        matrix = self.dataset.splits.get(split).complete_matrix
        embeddings = self._compute_embeddings(matrix, max_samples, **kwargs)

        return embeddings

# def enforcedataset(func):
#     @functools.wraps(func)
#     def wrapper(self, *args, **kwargs):
#         if not hasattr(self, "dataset") or self.dataset is None:
#             raise Exception("The evaluator must be updated with a dataset.")
#         return func(self, *args, **kwargs)
#
#     return wrapper
#
#
# class ModelEvaluator:
#     def __init__(self, recall_steps=[5, 20, 50], ndcg_k=100) -> None:
#         self.recall_steps = recall_steps
#         self.ndcg_k = ndcg_k
#         self.results: DataFrame = None
#         self.dataset: Dataset = None
#
#     def update_dataset(self, dataset: Dataset) -> None:
#         if not isinstance(dataset, Dataset):
#             raise Exception(
#                 "The data must be an instance of the dataset class."
#             )
#
#         self.dataset = dataset
#
#     @classmethod
#     def _recall(cls, argsort_mask, argsort_idx, X_true_binary):
#         K = argsort_mask.sum()
#         # an array of row number indices
#         rows_idx = jnp.arange(X_true_binary.shape[0])[:, jnp.newaxis]
#         # increase col indices by 1 to make zero index available as a null value
#         argsort_idx = argsort_idx + 1
#         # keep only first K cols for each row, the rest set to zero
#         cols_idx = jnp.where(argsort_mask, argsort_idx, 0)
#         # create a binary matrix from indices
#         # the matrix has an increased dimension by 1 to hold the increased indices
#         X_pred_shape = (X_true_binary.shape[0], X_true_binary.shape[1] + 1)
#         X_pred_binary = jnp.zeros(X_pred_shape, dtype=bool)
#         X_pred_binary = X_pred_binary.at[rows_idx, cols_idx].set(True)
#         # remove the first column that holds unused indices
#         X_pred_binary = X_pred_binary[:, 1:]
#         tmp = (
#             jnp.logical_and(X_true_binary, X_pred_binary).sum(axis=1)
#         ).astype(jnp.float32)
#         # divide a sum of the agreed indices by a sum of the true indices
#         # to avoid dividing by zero, use the K value as a minimum
#         recall = tmp / jnp.minimum(K, X_true_binary.sum(axis=1))
#         return recall
#
#     @classmethod
#     def _ndcg(cls, ndcg_mask, argsort_idx, X_true):
#         K = argsort_idx.shape[1]
#         rows_idx = jnp.arange(X_true.shape[0])[:, jnp.newaxis]
#         tp = 1.0 / jnp.log2(jnp.arange(2, K + 2))
#         # the slowest part ...
#         DCG = (X_true[rows_idx, argsort_idx] * tp).sum(axis=1)
#         IDCG = jnp.where(ndcg_mask, tp, 0).sum(axis=1)
#         return DCG / IDCG
#
#     @classmethod
#     def _partial_sort(cls, X_pred, k):
#         rows_idx = np.arange(X_pred.shape[0])[:, np.newaxis]
#         # put indices lower than K on the left side of array
#         idx_topk_part = np.argpartition(-X_pred, k, axis=1)
#         # select unordered top-K predictions
#         topk_part = X_pred[rows_idx, idx_topk_part[:, :k]]
#         # sort selected predictions by their value
#         idx_part = np.argsort(-topk_part, axis=1)
#         # select ordered indices of predictions
#         idx_topk = idx_topk_part[rows_idx, idx_part]
#         return idx_topk
#
#     def get_prediction_eval(self, X_pred, X_true):
#         recall_k = max(self.recall_steps)
#         max_k = max(recall_k, self.ndcg_k)
#
#         X_true_binary = X_true > 0
#         X_true_non_zero = X_true_binary.sum(axis=1)
#
#         # the slowest part ...
#         if default_backend() == "cpu":
#             # apply numpy argpartition and sort the rest
#             # at this momement, there is no JAX variant of argpartition
#             argsort_idx = self._partial_sort(X_pred, max_k)
#         else:
#             # sort predictions from the highest and keep top-K only
#             argsort_idx = jnp.argsort(-X_pred, axis=1)
#
#         # create a bin mask for each K starting with K true values
#         # [[T, T, F, F, ...], [T, T, T, T, F, ...]]
#         recall_mask = (
#             jnp.arange(recall_k)[:, jnp.newaxis] < jnp.array(self.recall_steps)
#         ).T
#         ndcg_mask = (
#             jnp.arange(self.ndcg_k)[:, jnp.newaxis]
#             < jnp.minimum(self.ndcg_k, X_true_non_zero)
#         ).T
#
#         # iter over the bin mask and push argsort indices as a static value
#         vmap_recall = vmap(self._recall, in_axes=(0, None, None), out_axes=0)
#         jitted_batch_recall = jit(vmap_recall)
#         recall_results = jitted_batch_recall(
#             recall_mask, argsort_idx[:, :recall_k], X_true_binary
#         )
#
#         jitted_ndcg = jit(self._ndcg)
#         ndcg_results = jitted_ndcg(
#             ndcg_mask, argsort_idx[:, : self.ndcg_k], X_true
#         )
#
#         # converting a device array to a numpy array we enfroce to wait for the results
#         # or it is possible to call block_until_ready()
#         data = {}
#         for i, k in enumerate(self.recall_steps):
#             data[f"recall@{k}"] = np.asarray(recall_results[i])
#
#         data[f"ndcg@{self.ndcg_k}"] = np.asarray(ndcg_results)
#
#         df = pd.DataFrame(data=data)
#
#         return df
#
#     def get_model_eval(self, model: Model, X, X_true):
#         X_pred = model.predict(X)
#         self.results = self.get_prediction_eval(X_pred, X_true)
#
#     @enforcedataset
#     def test_model_eval(self, model: Model):
#         self.get_model_eval(
#             model,
#             self.dataset.test_data_tr,
#             self.dataset.test_data_te.toarray(),
#         )
#
#     @enforcedataset
#     def vad_model_eval(self, model: Model):
#         self.get_model_eval(
#             model, self.dataset.vad_data_tr, self.dataset.vad_data_te.toarray()
#         )
#
#     @classmethod
#     def _print_metric(cls, metric, data):
#         print(
#             f"{metric}=%.5f (%.5f)"
#             % (
#                 np.mean(data),
#                 np.std(data) / np.sqrt(len(data)),
#             )
#         )
#
#     def print(self):
#         for metric, result in self.results.items():
#             for k, data in result.items():
#                 self._print_metric(f"{metric}@{k}", data)
#
#     def save(self, dir_path: Text):
#         for metric, results in self.results.items():
#             for k, data in results.items():
#                 file_name = f"{metric}-{k}.npy"
#                 file_path = os.path.join(dir_path, file_name)
#                 with open(file_path, "wb") as f:
#                     np.save(f, data)
#
#     def load(self, dir_path: Text):
#         file_paths = glob.glob(os.path.join(dir_path, "*.npy"))
#         for file_path in file_paths:
#             file_name = os.path.basename(file_path)
#             metric, k = file_name.split(".")[0].split("-")
#
#             if not self.results.get(metric):
#                 self.results[metric] = {}
#
#             with open(file_path, "rb") as f:
#                 self.results[metric][k] = np.load(f)
