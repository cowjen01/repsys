from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler


def get_precision_recall(X_predict: ndarray, X_true: ndarray, sort_indices: ndarray, k: int) -> Tuple[ndarray, ndarray]:
    row_indices = np.arange(X_predict.shape[0])[:, np.newaxis]

    X_true_bin = X_true > 0
    X_true_nonzero = (X_true > 0).sum(axis=1)

    X_predict_bin = np.zeros_like(X_predict, dtype=bool)
    X_predict_bin[row_indices, sort_indices[:, :k]] = True

    hits = (np.logical_and(X_true_bin, X_predict_bin).sum(axis=1)).astype(np.float32)

    precision = hits / k
    recall = hits / np.minimum(k, X_true_nonzero)

    return precision, recall


def get_ndcg(
    X_predict: ndarray,
    X_true: ndarray,
    sort_indices: ndarray,
    true_sort_indices: ndarray,
    k: int,
) -> ndarray:
    row_indices = np.arange(X_predict.shape[0])[:, np.newaxis]

    X_true_bin = X_true > 0
    X_true_nonzero = X_true_bin.sum(axis=1)

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (X_true[row_indices, sort_indices[:, :k]] * discount).sum(axis=1)

    mask = np.transpose(np.arange(k)[:, np.newaxis] < np.minimum(k, X_true_nonzero))
    idcg = np.where(mask, (X_true[row_indices, true_sort_indices] * discount), 0).sum(axis=1)

    return dcg / idcg


def get_coverage(X_predict: ndarray, sort_indices: ndarray, k: int) -> float:
    n_covered_items = len(np.unique(np.concatenate(sort_indices[:, :k])))
    n_items = X_predict.shape[1]

    return n_covered_items / n_items


def get_diversity(X_distances: ndarray, sort_indices: ndarray, k: int) -> ndarray:
    def f(idx):
        pairs = np.array(np.meshgrid(idx, idx)).T.reshape(-1, 2)
        dist = X_distances[pairs[:, 0], pairs[:, 1]]
        return dist.sum()

    vf = np.vectorize(f, signature="(n)->()")
    distances = vf(sort_indices[:, :k])

    return distances / (k * (k - 1))


def get_novelty(X_train: csr_matrix, sort_indices: ndarray, k: int) -> ndarray:
    popularity = np.asarray((X_train > 0).sum(axis=0)).squeeze() / X_train.shape[0]

    def f(idx):
        return np.sum(-np.log2(popularity[idx]))

    vf = np.vectorize(f, signature="(n)->()")
    novelty = vf(sort_indices[:, :k])
    max_novelty = -np.log2(1 / X_train.shape[0])

    return novelty / (k * max_novelty)


def get_error_metrics(X_predict: ndarray, X_true: ndarray) -> Tuple[float, float, float]:
    diff = X_true - X_predict

    mae = np.abs(diff).mean(axis=1)
    mse = np.square(diff).mean(axis=1)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def get_item_pop(X_predict: ndarray) -> ndarray:
    scaler = MinMaxScaler()

    popularity = X_predict.sum(axis=0).reshape(-1, 1)
    popularity = scaler.fit_transform(popularity)
    popularity = popularity.reshape(1, -1).squeeze()

    return popularity


def get_plt(sort_indices: ndarray, long_tail_items: ndarray, k: int) -> ndarray:
    def f(idx):
        return len(np.intersect1d(long_tail_items, idx, assume_unique=True))

    vf = np.vectorize(f, signature="(n)->()")
    plt = vf(sort_indices[:, :k])

    return plt / k


def get_clt(sort_indices: ndarray, long_tail_items: ndarray, k: int) -> float:
    covered_items = np.unique(np.concatenate(sort_indices[:, :k]))
    tail_covered = len(np.intersect1d(covered_items, long_tail_items, assume_unique=True))

    return tail_covered / long_tail_items.shape[0]
