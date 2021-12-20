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
    ItemID,
    String,
    Tags,
    UserID,
    Rating,
    find_column,
    filter_columns,
)
from repsys.dataset.validation import validate_dataset

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
    def get_user_id(self, user_idx) -> int:
        return self._idx2user.get(user_idx)

    @enforce_fitted
    def get_item_id(self, item_idx) -> int:
        return self._idx2item.get(item_idx)

    @enforce_fitted
    def get_user_index(self, user_id) -> int:
        return self._user2idx.get(user_id)

    @enforce_fitted
    def get_item_index(self, item_id) -> int:
        return self._item2idx.get(item_id)

    @classmethod
    def _build_train_matrix(cls, tp):
        # internal indexes start from zero
        n_users = tp["user_idx"].max() + 1
        n_items = tp["item_idx"].max() + 1
        rows, cols = tp["user_idx"], tp["item_idx"]

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

        start_idx = min(tp_tr["user_idx"].min(), tp_te["user_idx"].min())
        end_idx = max(tp_tr["user_idx"].max(), tp_te["user_idx"].max())

        rows_tr, cols_tr = tp_tr["user_idx"] - start_idx, tp_tr["item_idx"]
        rows_te, cols_te = tp_te["user_idx"] - start_idx, tp_te["item_idx"]

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

    def _update_data(self, splits, items: DataFrame, user2idx: Dict[int, int], item2idx: Dict[int, int]):
        logger.info("Updating data ...")

        item_dtypes = self.item_dtypes()

        self._raw_train_data = splits[0]
        self._raw_vad_data = splits[1]
        self._raw_test_data = splits[2]

        self._user2idx = user2idx
        self._item2idx = item2idx

        self._idx2user = {y: x for x, y in user2idx.items()}
        self._idx2item = {y: x for x, y in item2idx.items()}

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

        self.vad_users = list(self._user2idx.keys())

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

        item_id_col = find_column(interact_dtypes, ItemID)
        user_id_col = find_column(interact_dtypes, UserID)
        rating_col = find_column(interact_dtypes, Rating)

        if rating_col:
            rating: Rating = interact_dtypes[rating_col]
            interacts = interacts[interacts[rating_col] >= rating.bin_threshold]

        logger.info("Splitting interactions ...")

        splitter = DatasetSplitter(user_id_col, item_id_col)
        train_split, vad_split, test_split = splitter.split(interacts)

        user_ids = train_split.users.append(vad_split.users).append(
            test_split.users
        )

        item_ids = pd.unique(train_split.train_data[item_id_col])

        user2idx = dict((uid, i) for (i, uid) in enumerate(user_ids))
        item2idx = dict((sid, i) for (i, sid) in enumerate(item_ids))

        def reindex(df):
            user_idx = list(map(lambda x: user2idx[x], df[user_id_col]))
            item_idx = list(map(lambda x: item2idx[x], df[item_id_col]))

            return pd.DataFrame(
                data={"user_idx": user_idx, "item_idx": item_idx},
                columns=["user_idx", "item_idx"],
            )

        train_data = reindex(train_split.train_data)

        vad_data_tr = reindex(vad_split.train_data)
        vad_data_te = reindex(vad_split.test_data)

        test_data_tr = reindex(test_split.train_data)
        test_data_te = reindex(test_split.test_data)

        splits = (
            train_data,
            (vad_data_tr, vad_data_te),
            (test_data_tr, test_data_te),
        )

        # keep only columns defined in the dtypes
        items = items[item_dtypes.keys()]
        # set index column as an index
        items = items.set_index(find_column(item_dtypes, ItemID))
        # filter only items included in the training data
        items = items[items.index.isin(item_ids)]

        self._update_data(splits, items, user2idx, item2idx)

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

    def _load_idx_data(self, file_name: Text):
        data = dict()
        with open(os.path.join(tmp_dir_path(), file_name), "r") as f:
            for i, line in enumerate(f):
                data[int(line.strip())] = i

        return data

    def _save_idx_data(self, data, file_name: Text):
        with open(os.path.join(tmp_dir_path(), file_name), "w") as f:
            for sid in data.keys():
                f.write("%s\n" % sid)

    def load(self, path: Text):
        create_tmp_dir()

        unzip_dir(path, tmp_dir_path())

        item_dtypes = self.item_dtypes()
        item_id_col = find_column(item_dtypes, ItemID)

        train_data_path = os.path.join(tmp_dir_path(), "train.csv")
        train_data = pd.read_csv(train_data_path)

        items_data_path = os.path.join(tmp_dir_path(), "items.csv")
        items = pd.read_csv(items_data_path, index_col=item_id_col)

        vad_data = self._load_tr_te_data("vad")
        test_data = self._load_tr_te_data("test")

        item2ids = self._load_idx_data("item_index.txt")
        user2idx = self._load_idx_data("user_index.txt")

        splits = train_data, vad_data, test_data

        try:
            self._update_data(splits, items, user2idx, item2ids)
        finally:
            remove_tmp_dir()

    @enforce_fitted
    def save(self, path: Text):
        create_tmp_dir()

        train_data_path = os.path.join(tmp_dir_path(), "train.csv")
        self._raw_train_data.to_csv(train_data_path, index=False)

        items_data_path = os.path.join(tmp_dir_path(), "items.csv")
        self.items.to_csv(items_data_path, index=True)

        self._save_tr_te_data(self._raw_vad_data, "vad")
        self._save_tr_te_data(self._raw_test_data, "test")

        self._save_idx_data(self._item2idx, "item_index.txt")
        self._save_idx_data(self._user2idx, "user_index.txt")

        zip_dir(path, tmp_dir_path())

        remove_tmp_dir()

    def __str__(self):
        return f"Dataset '{self.name()}'"
