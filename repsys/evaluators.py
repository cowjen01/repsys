from typing import Tuple, Dict, Optional

import pandas as pd
import pymde
from jax import numpy as jnp, vmap, jit, default_backend
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from repsys.dataset import Dataset
from repsys.helpers import *
from repsys.model import Model


def embeddings_to_df(embeds: ndarray, ids: ndarray) -> DataFrame:
    df = pd.DataFrame({'id': ids, 'x': embeds[:, 0], 'y': embeds[:, 1]})
    return df


def results_to_df(results: Dict[str, ndarray], ids: List[str]) -> DataFrame:
    df = pd.DataFrame({'id': ids, **results})
    return df


def write_df(df: DataFrame, file_name: str) -> None:
    df.to_csv(os.path.join(tmp_dir_path(), file_name), index=False)


def read_df(file_path: str) -> DataFrame:
    return pd.read_csv(file_path, dtype={'id': str}).set_index('id')


class DatasetEvaluator:
    def __init__(self, dataset: Dataset, seed: int = 1234, verbose: bool = True):
        self._dataset = dataset
        self.verbose = verbose
        self.seed = seed
        self.item_embeddings = {}
        self.user_embeddings = {}

    def _get_input_data(self, split: str) -> csr_matrix:
        return self._dataset.splits.get(split).complete_matrix

    def _get_embeddings(self, matrix: csr_matrix, max_samples: int = 10000, method: str = 'pymde', **kwargs) -> Tuple[
        ndarray, ndarray]:
        set_seed(self.seed)
        index_perm = np.random.permutation(matrix.shape[0])
        index_perm = index_perm[: max_samples]
        matrix = matrix[index_perm]

        if method == 'pymde':
            pymde.seed(self.seed)
            mde = pymde.preserve_neighbors(matrix, init='random', n_neighbors=10, verbose=self.verbose, **kwargs)
            embeddings = mde.embed(verbose=self.verbose, max_iter=1000, memory_size=50, eps=1e-6)
            embeddings = embeddings.cpu().numpy()
        else:
            pca = PCA(n_components=50).fit_transform(matrix.toarray())
            embeddings = TSNE(n_iter=1500, n_components=2, metric='cosine', init='random', verbose=2).fit_transform(pca)

        return embeddings, index_perm

    def compute_item_embeddings(self, split: str = 'train', **kwargs) -> None:
        matrix = self._get_input_data(split).T
        embeds, indices = self._get_embeddings(matrix, **kwargs)
        ids = np.vectorize(self._dataset.item_index_to_id)(indices)
        self.item_embeddings[split] = embeddings_to_df(embeds, ids)

    def compute_user_embeddings(self, split: str = 'train', **kwargs) -> None:
        matrix = self._get_input_data(split)
        embeds, indices = self._get_embeddings(matrix, **kwargs)
        ids = np.vectorize(self._dataset.user_index_iterator(split))(indices)
        self.user_embeddings[split] = embeddings_to_df(embeds, ids)

    def compute_embeddings(self, split: str = 'train', **kwargs):
        self.compute_user_embeddings(split, **kwargs)
        self.compute_item_embeddings(split, **kwargs)

    @tmpdir_provider
    def save(self, checkpoints_dir: str) -> None:
        for split, df in self.item_embeddings.items():
            write_df(df, f'items-{split}.csv')

        for split, df in self.user_embeddings.items():
            write_df(df, f'users-{split}.csv')

        filename = f'dataset-eval-{current_ts()}.zip'
        file_path = os.path.join(checkpoints_dir, filename)

        zip_dir(file_path, tmp_dir_path())

    @tmpdir_provider
    def load(self, checkpoints_dir: str) -> None:
        checkpoints = find_checkpoints(checkpoints_dir, 'dataset-eval-*.zip')

        if not checkpoints:
            return

        file_path = checkpoints[0]
        unzip_dir(file_path, tmp_dir_path())

        for split in ['train', 'validation', 'test']:
            items_path = os.path.join(tmp_dir_path(), f'items-{split}.csv')
            if os.path.isfile(items_path):
                self.item_embeddings[split] = read_df(items_path)

            users_path = os.path.join(tmp_dir_path(), f'users-{split}.csv')
            if os.path.isfile(users_path):
                self.user_embeddings[split] = read_df(users_path)


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


def partial_sort(x_predict: csr_matrix, k: int):
    row_indices = np.arange(x_predict.shape[0])[:, np.newaxis]
    # put indices lower than K on the left side of array
    top_k_indices = np.argpartition(x_predict, k, axis=1)[:, :k]
    # select unordered top-K predictions
    top_k_predicts = x_predict[row_indices, top_k_indices]
    # sort selected predictions by their value
    sorted_indices = np.argsort(top_k_predicts, axis=1)
    # get indices of the highest predictions
    return top_k_indices[row_indices, sorted_indices]


class ModelEvaluator:
    def __init__(self, dataset: Dataset, recall_steps: List[int] = None, ndcg_k: int = 100):
        if recall_steps is None:
            recall_steps = [20, 50]

        self.recall_steps = recall_steps
        self.ndcg_k = ndcg_k
        self.evaluated_models: List[str] = []
        self._dataset = dataset
        self._user_results: Dict[str, List[DataFrame]] = {}

        self.summary_metrics = [f"recall@{k}" for k in self.recall_steps] + [f"ndcg@{self.ndcg_k}"]
        self.user_metrics = self.summary_metrics + ["mae", "mse", "rmse"]

    def compute_user_metrics(self, x_predict: ndarray, x_true: ndarray) -> Dict[str, ndarray]:
        max_recall_k = max(self.recall_steps)
        # use the highest K of the configured steps
        max_k = max(max_recall_k, self.ndcg_k)

        x_true = np.clip(x_true, -1, 1)
        x_predict = np.clip(x_predict, -1, 1)
        x_true_binary = (x_true > 0)
        x_true_nonzero = x_true_binary.sum(axis=1)

        if default_backend() == "cpu":
            # at this time there is no jax implementation of the arg partition
            # at cpu we switch to the classical numpy for a moment to sort the array
            sort_indices = partial_sort(-x_predict, max_k)
        else:
            # on gpu sorting of an array is much faster, so we sort it all
            # once jax arg partition will be implemented, it should appear here
            sort_indices = jnp.argsort(-x_predict, axis=1)

        # create a binary mask for each K starting with K true values nad following false values
        # this way we lately mask only the top-k values and ignore the rest
        # [[T, T, F, F, ...], [T, T, T, T, F, ...]]
        recall_k_mask = (jnp.arange(max_recall_k)[:, jnp.newaxis] < jnp.array(self.recall_steps)).T
        ndcg_mask = (jnp.arange(self.ndcg_k)[:, jnp.newaxis] < jnp.minimum(self.ndcg_k, x_true_nonzero)).T

        # iterate over the binary mask and push sorted indices as a static value
        vmap_recall = vmap(recall, in_axes=(0, None, None), out_axes=0)
        jitted_recall = jit(vmap_recall)
        top_recall_indices = sort_indices[:, :max_recall_k]
        recall_results = jitted_recall(recall_k_mask, top_recall_indices, x_true_binary)

        jitted_ndcg = jit(ndcg)
        ndcg_results = jitted_ndcg(ndcg_mask, sort_indices[:, :self.ndcg_k], x_true)

        diff = x_true - x_predict
        mae = jnp.abs(diff).mean(axis=1)
        mse = jnp.square(diff).mean(axis=1)
        rmse = jnp.sqrt(mse)

        # converting the device array to a numpy array we enforce to wait for the results
        # also it is possible to call block_until_ready()
        results = {}
        for i, k in enumerate(self.recall_steps):
            results[f"recall@{k}"] = np.asarray(recall_results[i])

        results[f"ndcg@{self.ndcg_k}"] = np.asarray(ndcg_results)

        results["mae"] = mae
        results["mse"] = mse
        results["rmse"] = rmse

        return results

    def print(self) -> None:
        for model in self.evaluated_models:
            print(f'MODEL {model.upper()}:')
            user_results = self._user_results.get(model)
            print('User Metrics:')
            for i, df in enumerate(user_results):
                print(f'History {i}:')
                print(df.describe())

    def get_user_results(self, model_name: str, history: int = 0) -> Optional[DataFrame]:
        results = self._user_results.get(model_name)
        if not results or len(results) - 1 < history:
            return None

        return results[history]

    def get_eval_summary(self, model_name: str, history: int = 0) -> Optional[Dict[str, float]]:
        user_results = self.get_user_results(model_name, history)

        user_summary = {}
        if user_results is not None:
            user_summary = user_results.mean().to_dict()
            user_summary = {metric: user_summary[metric] for metric in self.summary_metrics}

        summary = {**user_summary}

        return summary

    def evaluate(self, model: Model, split: str = 'validation'):
        test_split = self._dataset.splits.get(split)
        x_true = test_split.holdout_matrix.toarray()
        x_predict = model.predict(test_split.train_matrix)

        user_results = self.compute_user_metrics(x_predict, x_true)
        user_ids = list(test_split.user_index.keys())
        df = results_to_df(user_results, user_ids)

        if model.name() not in self.evaluated_models:
            self.evaluated_models.append(model.name())
            self._user_results[model.name()] = [df]
        else:
            self._user_results.get(model.name()).append(df)

        self.print()

    @tmpdir_provider
    def _save_latest_eval(self, model_name: str, checkpoints_dir: str):
        user_results = self._user_results.get(model_name)[-1]
        write_df(user_results, 'user-metrics.csv')
        filename = f'model-eval-{model_name}-{current_ts()}.zip'
        file_path = os.path.join(checkpoints_dir, filename)
        zip_dir(file_path, tmp_dir_path())

    def save(self, checkpoints_dir: str) -> None:
        for model in self.evaluated_models:
            self._save_latest_eval(model, checkpoints_dir)

    @tmpdir_provider
    def _load_model_eval(self, model_name: str, zip_path: str):
        unzip_dir(zip_path, tmp_dir_path())
        user_results_path = os.path.join(tmp_dir_path(), 'user-metrics.csv')
        user_results = read_df(user_results_path)
        self._user_results[model_name].append(user_results)

    def load(self, checkpoints_dir: str, models: List[str], history: int = 1) -> None:
        self.evaluated_models = []
        for model in models:
            pattern = f'model-eval-{model}-*.zip'
            checkpoints = find_checkpoints(checkpoints_dir, pattern, history)

            if checkpoints:
                self._user_results[model] = []
                self.evaluated_models.append(model)

            for zip_path in checkpoints:
                self._load_model_eval(model, zip_path)
