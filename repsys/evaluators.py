from typing import Tuple, Dict, Optional, Any

import pandas as pd
import pymde
from jax import numpy as jnp, vmap, jit, default_backend
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from repsys.dataset import Dataset
from repsys.helpers import *
from repsys.metrics import recall, ndcg
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
    def __init__(self, dataset: Dataset, split: str = 'train', seed: int = 1234, verbose: bool = True):
        self._dataset = dataset
        self._split = split
        self._verbose = verbose
        self._seed = seed
        self._tsne = TSNE(n_iter=1500, n_components=2, metric='cosine', init='random', verbose=self._verbose)
        self._pca = PCA(n_components=50)
        self.item_embeddings: Optional[DataFrame] = None
        self.user_embeddings: Optional[DataFrame] = None

    def _sample_data(self, X: ndarray, max_samples: int) -> Tuple[ndarray, ndarray]:
        set_seed(self._seed)
        indices = np.random.permutation(X.shape[0])
        indices = indices[: max_samples]
        return X[indices], indices

    def _pymde_embeddings(self, X: Any) -> ndarray:
        pymde.seed(self._seed)
        mde = pymde.preserve_neighbors(X, init='random', n_neighbors=10, verbose=self._verbose)
        embeddings = mde.embed(verbose=self._verbose, max_iter=1000, memory_size=50, eps=1e-6)
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def _tsne_embeddings(self, X: Any) -> ndarray:
        if issparse(X):
            X = X.toarray()
        if X.shape[1] > 50:
            X = self._pca.fit_transform(X)
        embeddings = self._tsne.fit_transform(X)
        return embeddings

    def compute_embeddings(self, method: str = 'pymde', max_samples: int = None):
        X = self._dataset.splits.get(self._split).complete_matrix

        if method == 'pymde':
            user_embeds = self._pymde_embeddings(X)
            item_embeds = self._pymde_embeddings(X.T)
        elif method == 'tsne':
            user_embeds = self._tsne_embeddings(X)
            item_embeds = self._tsne_embeddings(X.T)
        else:
            user_embeds, item_embeds = self._dataset.compute_embeddings(X)
            if user_embeds.shape[1] > 2:
                user_embeds = self._pymde_embeddings(user_embeds)
            if item_embeds.shape[1] > 2:
                item_embeds = self._pymde_embeddings(item_embeds)

        if max_samples is not None:
            user_embeds, user_indices = self._sample_data(user_embeds, max_samples)
            item_embeds, item_indices = self._sample_data(item_embeds, max_samples)
        else:
            user_indices = np.arange(user_embeds.shape[0])
            item_indices = np.arange(item_embeds.shape[0])

        user_ids = np.vectorize(self._dataset.user_index_iterator(self._split))(user_indices)
        item_ids = np.vectorize(self._dataset.item_index_to_id)(item_indices)

        self.user_embeddings = embeddings_to_df(user_embeds, user_ids)
        self.item_embeddings = embeddings_to_df(item_embeds, item_ids)

    @tmpdir_provider
    def save(self, checkpoints_dir: str) -> None:
        write_df(self.user_embeddings, f'user-embeds.csv')
        write_df(self.item_embeddings, f'item-embeds.csv')

        filename = f'dataset-eval-{self._split}-{current_ts()}.zip'
        file_path = os.path.join(checkpoints_dir, filename)

        zip_dir(file_path, tmp_dir_path())

    @tmpdir_provider
    def load(self, checkpoints_dir: str) -> None:
        pattern = f'dataset-eval-{self._split}-*.zip'
        checkpoints = find_checkpoints(checkpoints_dir, pattern)

        if not checkpoints:
            return

        file_path = checkpoints[0]
        unzip_dir(file_path, tmp_dir_path())

        items_path = os.path.join(tmp_dir_path(), 'item-embeds.csv')
        self.item_embeddings = read_df(items_path)

        users_path = os.path.join(tmp_dir_path(), 'user-embeds.csv')
        self.user_embeddings = read_df(users_path)


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

    @staticmethod
    def _partial_sort(x_predict: csr_matrix, k: int):
        row_indices = np.arange(x_predict.shape[0])[:, np.newaxis]
        # put indices lower than K on the left side of array
        top_k_indices = np.argpartition(x_predict, k, axis=1)[:, :k]
        # select unordered top-K predictions
        top_k_predicts = x_predict[row_indices, top_k_indices]
        # sort selected predictions by their value
        sorted_indices = np.argsort(top_k_predicts, axis=1)
        # get indices of the highest predictions
        return top_k_indices[row_indices, sorted_indices]

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
            sort_indices = self._partial_sort(-x_predict, max_k)
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
            print('-----------------------')
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
