import logging
from typing import Tuple, Dict, Optional, Any

import pandas as pd
import pymde
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from repsys.dataset import Dataset
from repsys.helpers import *
from repsys.metrics import get_pr, get_ndcg, get_accuracy_metrics, get_coverage, get_diversity
from repsys.model import Model

logger = logging.getLogger(__name__)


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


def sort_partially(X: ndarray, k: int) -> ndarray:
    row_indices = np.arange(X.shape[0])[:, np.newaxis]
    top_k_indices = np.argpartition(X, k, axis=1)[:, :k]
    top_k_predicts = X[row_indices, top_k_indices]
    sorted_indices = np.argsort(top_k_predicts, axis=1)
    return top_k_indices[row_indices, sorted_indices]


class ModelEvaluator:
    def __init__(self, dataset: Dataset, rp_k: List[int] = None, ndcg_k: List[int] = None,
                 coverage_k: List[int] = None):
        if rp_k is None:
            rp_k = [20, 50]

        if ndcg_k is None:
            ndcg_k = [100]

        if coverage_k is None:
            coverage_k = [20, 50]

        self.rp_k = rp_k
        self.ndcg_k = ndcg_k
        self.coverage_k = coverage_k
        self.diversity_k = [5, 10]

        self.evaluated_models: List[str] = []

        self._dataset = dataset
        self._user_results: Dict[str, List[DataFrame]] = {}
        self._item_results: Dict[str, List[DataFrame]] = {}

        recall_metrics = [f"recall@{k}" for k in self.rp_k]
        precision_metrics = [f"precision@{k}" for k in self.rp_k]
        ndcg_metrics = [f"ndcg@{k}" for k in self.ndcg_k]
        coverage_metrics = [f"coverage@{k}" for k in self.coverage_k]
        diversity_metrics = [f"diversity@{k}" for k in self.diversity_k]

        self.summary_metrics = recall_metrics + precision_metrics + ndcg_metrics + coverage_metrics + diversity_metrics
        self.user_metrics = recall_metrics + precision_metrics + ndcg_metrics + diversity_metrics + ["mae", "mse", "rmse"]

    def compute_metrics(self, X_predict: ndarray, X_true: ndarray) -> Tuple[Dict[str, ndarray], Dict[str, float]]:
        max_k = max(max(self.rp_k), max(self.ndcg_k))

        logger.info(f"Sorting predictions for maximal K={max_k}")
        predict_sort = sort_partially(-X_predict, k=max_k)
        true_sort = sort_partially(-X_true, k=max_k)

        logger.info("Computing precision and recall")
        user_results = {}
        for k in self.rp_k:
            precision, recall = get_pr(X_predict, X_true, predict_sort, k)
            user_results[f"precision@{k}"] = precision
            user_results[f"recall@{k}"] = recall

        logger.info("Computing NDCG")
        for k in self.ndcg_k:
            ndcg = get_ndcg(X_predict, X_true, predict_sort, true_sort, k)
            user_results[f"ndcg@{k}"] = ndcg

        logger.info("Computing diversity")
        X_train = self._dataset.get_train_data()
        for k in self.diversity_k:
            user_results[f"diversity@{k}"] = get_diversity(X_train, predict_sort, k)

        logger.info("Computing MAE, MSE and RMSE")
        mae, mse, rmse = get_accuracy_metrics(X_predict, X_true)
        user_results["mae"], user_results["mse"], user_results["rmse"] = mae, mse, rmse

        logger.info("Computing coverage")
        item_results = {}
        for k in self.coverage_k:
            item_results[f"coverage@{k}"] = get_coverage(X_predict, predict_sort, k)

        return user_results, item_results

    def print(self) -> None:
        for model in self.evaluated_models:
            print(f'\n{model.upper()} MODEL:')
            print('-----------------------------')
            user_results = self._user_results.get(model)
            item_results = self._item_results.get(model)
            print('User Metrics:')
            for i, df in enumerate(user_results):
                print(f'- history {i + 1}:')
                print(df.describe())
                print('\n')
            print('Item Metrics:')
            for i, df in enumerate(item_results):
                print(f'- history {i + 1}:')
                print(df.describe())
                print('\n')

    def get_user_results(self, model_name: str, history: int = 0) -> Optional[DataFrame]:
        results = self._user_results.get(model_name)
        if not results or len(results) - 1 < history:
            return None

        return results[history]

    def get_item_results(self, model_name: str, history: int = 0) -> Optional[DataFrame]:
        results = self._item_results.get(model_name)
        if not results or len(results) - 1 < history:
            return None

        return results[history]

    def summary(self, model_name: str, history: int = 0) -> Optional[Dict[str, float]]:
        user_results = self.get_user_results(model_name, history)
        item_results = self.get_item_results(model_name, history)

        user_summary = {}
        if user_results is not None:
            user_summary = user_results.mean().to_dict()

        item_summary = {}
        if item_results is not None:
            item_summary = item_results.mean().to_dict()

        summary = {**user_summary, **item_summary}

        return summary

    def evaluate(self, model: Model, split: str = 'validation'):
        test_split = self._dataset.splits.get(split)
        x_true = test_split.holdout_matrix.toarray()

        logger.info("Computing predictions")
        x_predict = model.predict(test_split.train_matrix)

        user_results, item_results = self.compute_metrics(x_predict, x_true)
        user_ids = list(test_split.user_index.keys())
        user_df = results_to_df(user_results, user_ids)

        item_df = pd.DataFrame(item_results, index=[0])

        if model.name() not in self.evaluated_models:
            self.evaluated_models.append(model.name())
            self._user_results[model.name()] = [user_df]
            self._item_results[model.name()] = [item_df]
        else:
            self._user_results.get(model.name()).append(user_df)
            self._item_results.get(model.name()).append(item_df)

    @tmpdir_provider
    def _save_latest_eval(self, model_name: str, checkpoints_dir: str):
        user_results = self._user_results.get(model_name)[-1]
        item_results = self._item_results.get(model_name)[-1]

        write_df(user_results, 'user-metrics.csv')
        write_df(item_results, 'item-metrics.csv')

        filename = f'model-eval-{model_name}-{current_ts()}.zip'
        file_path = os.path.join(checkpoints_dir, filename)
        zip_dir(file_path, tmp_dir_path())

    def save(self, checkpoints_dir: str) -> None:
        for model in self.evaluated_models:
            self._save_latest_eval(model, checkpoints_dir)

    @tmpdir_provider
    def _load_model_eval(self, model_name: str, zip_path: str):
        unzip_dir(zip_path, tmp_dir_path())

        user_path = os.path.join(tmp_dir_path(), 'user-metrics.csv')
        item_path = os.path.join(tmp_dir_path(), 'item-metrics.csv')

        user_results = read_df(user_path)
        item_results = pd.read_csv(item_path)

        self._user_results[model_name].append(user_results)
        self._item_results[model_name].append(item_results)

    def load(self, checkpoints_dir: str, models: List[str], history: int = 1) -> None:
        self.evaluated_models = []
        for model in models:
            pattern = f'model-eval-{model}-*.zip'
            checkpoints = find_checkpoints(checkpoints_dir, pattern, history)

            if checkpoints:
                self._user_results[model] = []
                self._item_results[model] = []
                self.evaluated_models.append(model)

            for zip_path in checkpoints:
                self._load_model_eval(model, zip_path)


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
