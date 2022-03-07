from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances


def get_pr(X_predict: ndarray, X_true: ndarray, predict_sort: ndarray, k: int) -> Tuple[ndarray, ndarray]:
    row_indices = np.arange(X_predict.shape[0])[:, np.newaxis]

    X_true_bin = X_true > 0
    X_true_nonzero = (X_true > 0).sum(axis=1)

    X_predict_bin = np.zeros_like(X_predict, dtype=bool)
    X_predict_bin[row_indices, predict_sort[:, :k]] = True

    hits = (np.logical_and(X_true_bin, X_predict_bin).sum(axis=1)).astype(np.float32)

    precision = hits / k
    recall = hits / np.minimum(k, X_true_nonzero)

    return precision, recall


def get_ndcg(X_predict: ndarray, X_true: ndarray, predict_sort: ndarray, true_sort: ndarray, k: int) -> ndarray:
    row_indices = np.arange(X_predict.shape[0])[:, np.newaxis]

    X_true_bin = X_true > 0
    X_true_nonzero = X_true_bin.sum(axis=1)

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (X_true[row_indices, predict_sort[:, :k]] * discount).sum(axis=1)

    mask = np.transpose(np.arange(k)[:, np.newaxis] < np.minimum(k, X_true_nonzero))
    idcg = np.where(mask, (X_true[row_indices, true_sort] * discount), 0).sum(axis=1)

    return dcg / idcg


def get_coverage(X_predict: ndarray, predict_sort: ndarray, k: int) -> float:
    n_covered_items = len(np.unique(np.concatenate(predict_sort[:, :k])))
    n_items = X_predict.shape[1]

    return n_covered_items / n_items


def get_diversity(X_train: csr_matrix, predict_sort: ndarray, k: int) -> ndarray:
    X_train = X_train.T
    X_distances = pairwise_distances(X_train, metric='cosine')

    def f(idx):
        pairs = np.array(np.meshgrid(idx, idx)).T.reshape(-1, 2)
        dist = X_distances[pairs[:, 0], pairs[:, 1]]
        return dist.sum()

    vf = np.vectorize(f, signature='(n)->()')
    distances = vf(predict_sort[:, :k])

    return distances / (k * (k - 1))


def get_accuracy_metrics(X_predict, X_true):
    diff = X_true - X_predict
    mae = np.abs(diff).mean(axis=1)
    mse = np.square(diff).mean(axis=1)
    rmse = np.sqrt(mse)

    return mae, mse, rmse
