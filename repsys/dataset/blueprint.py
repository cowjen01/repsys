# credits: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

import logging
import os
from abc import ABC, abstractmethod
from typing import Text, Dict
from pandas import DataFrame
from scipy import sparse
import numpy as np
import pandas as pd
import functools

from repsys.utils import (
    create_tmp_dir,
    tmp_dir_path,
    remove_tmp_dir,
    unzip_dir,
    zip_dir,
)
from repsys.dataset.splitter import DatasetSplitter
from repsys.dataset.dtypes import (
    DataType,
    ItemIndex,
    String,
    Tags,
    UserIndex,
    Rating,
    find_column,
    filter_columns,
)
from repsys.dataset.validators import validate_dataset

logger = logging.getLogger(__name__)


def enforce_fitted(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise Exception("The dataset has not been fitted yet.")
        return func(self, *args, **kwargs)

    return wrapper


class Dataset(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def item_dtypes(self) -> Dict[Text, DataType]:
        pass

    @abstractmethod
    def interact_dtypes(self) -> Dict[Text, DataType]:
        pass

    @abstractmethod
    def load_items(self) -> DataFrame:
        pass

    @abstractmethod
    def load_interacts(self) -> DataFrame:
        pass

    @enforce_fitted
    def _get_user_id(self, user_index) -> int:
        return self._user2id.get(user_index)

    @enforce_fitted
    def _get_item_id(self, item_index) -> int:
        return self._item2id.get(item_index)

    @enforce_fitted
    def _get_user_index(self, user_id) -> int:
        return self._id2user.get(user_id)

    @enforce_fitted
    def _get_item_index(self, item_id) -> int:
        return self._id2item.get(item_id)

    @classmethod
    def _build_train_matrix(cls, tp):
        # internal indexes start from zero
        n_users = tp["uid"].max() + 1
        n_items = tp["sid"].max() + 1
        rows, cols = tp["uid"], tp["sid"]

        # put one only at places, where uid and sid index meet
        data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype="float64",
            shape=(n_users, n_items),
        )

        return data, n_items

    @classmethod
    def _build_tr_te_matrix(cls, tp, n_items):
        tp_tr, tp_te = tp

        start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
        end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

        rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
        rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, n_items),
        )

        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, n_items),
        )

        return data_tr, data_te

    def _process_data(self, splits, items: DataFrame):
        item_dtypes = self.item_dtypes()

        self._raw_train_data = splits[0]
        self._raw_vad_data = splits[1]
        self._raw_test_data = splits[2]
        self._user2id, self._item2id = splits[3]

        self._id2user = {y: x for x, y in self._user2id.items()}
        self._id2item = {y: x for x, y in self._item2id.items()}

        logger.info("Building sparse matrices ...")

        self.train_data, self.n_items = self._build_train_matrix(
            self._raw_train_data
        )
        self.vad_data_tr, self.vad_data_te = self._build_tr_te_matrix(
            self._raw_vad_data, self.n_items
        )
        self.test_data_tr, self.test_data_te = self._build_tr_te_matrix(
            self._raw_test_data, self.n_items
        )

        self.vad_users = list(self._user2id.keys())

        logger.info("Processing items data ...")

        tags_columns = filter_columns(item_dtypes, Tags)
        string_columns = filter_columns(item_dtypes, String)

        self.tags = {}
        for tags_col in tags_columns:
            tags: Tags = item_dtypes[tags_col]
            self.tags[tags_col] = (
                items[tags_col]
                .dropna()
                .str.split(tags.sep, expand=True)
                .stack()
                .unique()
                .tolist()
            )
            items[tags_col] = items[tags_col].fillna("")

        for string_col in string_columns:
            items[string_col] = items[string_col].fillna("")

        self.items = items
        self._is_fitted = True

    def fit(self):
        logger.info("Loading dataset ...")

        items = self.load_items()
        item_dtypes = self.item_dtypes()
        interacts = self.load_interacts()
        interact_dtypes = self.interact_dtypes()

        logger.info("Validating dataset ...")

        validate_dataset(interacts, items, interact_dtypes, item_dtypes)

        item_index_col = find_column(interact_dtypes, ItemIndex)
        user_index_col = find_column(interact_dtypes, UserIndex)
        rating_col = find_column(interact_dtypes, Rating)

        if rating_col:
            rating: Rating = interact_dtypes[rating_col]
            interacts = interacts[interacts[rating_col] >= rating.bin_threshold]

        logger.info("Splitting interactions ...")

        splitter = DatasetSplitter(user_index_col, item_index_col)
        splits = splitter.split(interacts)

        item2id = splits[3][1]

        item_index_col = find_column(item_dtypes, ItemIndex)
        # keep only columns defined in the dtypes
        items = items[item_dtypes.keys()]
        # set index column as an index
        items = items.set_index(item_index_col)
        # filter only items included in the training data
        items = items[items.index.isin(item2id.keys())]

        self._process_data(splits, items)

    def _load_tr_te_data(self, split):
        data_tr_path = os.path.join(tmp_dir_path(), f"{split}_tr.csv")
        data_te_path = os.path.join(tmp_dir_path(), f"{split}_te.csv")

        return (
            pd.read_csv(data_tr_path),
            pd.read_csv(data_te_path),
        )

    def _save_tr_te_data(self, data, split):
        data_tr_path = os.path.join(tmp_dir_path(), f"{split}_tr.csv")
        data_te_path = os.path.join(tmp_dir_path(), f"{split}_te.csv")

        data[0].to_csv(data_tr_path, index=False)
        data[1].to_csv(data_te_path, index=False)

    def _load_dict_data(self, file_name: Text):
        data = dict()
        with open(os.path.join(tmp_dir_path(), file_name), "r") as f:
            for i, line in enumerate(f):
                data[line.strip()] = i

        return data

    def _save_dict_data(self, data, file_name: Text):
        with open(os.path.join(tmp_dir_path(), file_name), "w") as f:
            for sid in data.keys():
                f.write("%s\n" % sid)

    def load(self, path: Text):
        create_tmp_dir()

        unzip_dir(path, tmp_dir_path())

        train_data_path = os.path.join(tmp_dir_path(), "train.csv")
        train_data = pd.read_csv(train_data_path)

        items_data_path = os.path.join(tmp_dir_path(), "items.csv")
        items = pd.read_csv(items_data_path)

        vad_data = self._load_tr_te_data("vad")
        test_data = self._load_tr_te_data("test")

        item2id = self._load_dict_data("unique_sid.txt")
        user2id = self._load_dict_data("unique_uid.txt")

        splits = (train_data, vad_data, test_data, (user2id, item2id))

        self._process_data(splits, items)

        remove_tmp_dir()

    @enforce_fitted
    def save(self, path: Text):
        create_tmp_dir()

        train_data_path = os.path.join(tmp_dir_path(), "train.csv")
        self._raw_train_data.to_csv(train_data_path, index=False)

        items_data_path = os.path.join(tmp_dir_path(), "items.csv")
        self.items.to_csv(items_data_path, index=False)

        self._save_tr_te_data(self._raw_vad_data, "vad")
        self._save_tr_te_data(self._raw_test_data, "test")

        self._save_dict_data(self._item2id, "unique_sid.txt")
        self._save_dict_data(self._user2id, "unique_uid.txt")

        zip_dir(path, tmp_dir_path())

        remove_tmp_dir()

    def __str__(self):
        return f"Dataset '{self.name()}'"
