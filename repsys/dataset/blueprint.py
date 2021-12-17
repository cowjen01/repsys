# credits: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

import logging
from abc import ABC, abstractmethod
from typing import Text, Dict
from pandas import DataFrame
import pandas as pd
from scipy import sparse
import numpy as np

from repsys.dataset import DatasetSplitter, DatasetValidator
from repsys.dataset.dtypes import (
    DataType,
    DTypesParser,
    ItemIndex,
    String,
    Tags,
    UserIndex,
    Rating,
)

logger = logging.getLogger(__name__)


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
    def _build_tr_te_matrix(cls, tp_tr, tp_te, n_items):
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

    def fit(self):
        logger.info("Loading dataset ...")

        items = self.load_items()
        item_dtypes = self.item_dtypes()
        interacts = self.load_interacts()
        interact_dtypes = self.interact_dtypes()

        logger.info("Validating dataset ...")

        validator = DatasetValidator()
        validator.validate(interacts, items, interact_dtypes, item_dtypes)

        interacts_item_index = DTypesParser.find_first(
            interact_dtypes, ItemIndex
        )
        interacts_user_index = DTypesParser.find_first(
            interact_dtypes, UserIndex
        )
        rating_index = DTypesParser.find_first(interact_dtypes, Rating)

        if rating_index:
            rating: Rating = interact_dtypes[rating_index]
            interacts = interacts[interacts[rating_index] >= rating.threshold]

        logger.info("Splitting interactions ...")

        splitter = DatasetSplitter(interacts_user_index, interacts_item_index)
        (
            raw_train_data,
            raw_vad_data,
            raw_test_data,
            (user2id, item2id),
        ) = splitter.split(interacts)

        self.raw_data = {
            "train": raw_train_data,
            "vad": raw_vad_data,
            "test": raw_test_data,
        }

        logger.info("Building sparse matrices ...")

        self.train_data, self.n_items = self._build_train_matrix(raw_train_data)
        self.vad_data_tr, self.vad_data_te = self._build_tr_te_matrix(
            raw_vad_data[0], raw_vad_data[1], self.n_items
        )
        self.test_data_tr, self.test_data_te = self._build_tr_te_matrix(
            raw_test_data[0], raw_test_data[1], self.n_items
        )

        vad_users = pd.DataFrame(user2id.keys(), columns=['label'])
        vad_users['label'] = vad_users['label'].astype(str)
        vad_users["id"] = vad_users.index

        self.vad_users = vad_users

        logger.info("Processing items data ...")

        item_index = DTypesParser.find_first(item_dtypes, ItemIndex)
        tags_idxs = DTypesParser.find_all(item_dtypes, Tags)
        string_idxs = DTypesParser.find_all(item_dtypes, String)

        # set index column as an index
        items = items.set_index(item_index)
        # filter only items included in the training data
        items = items[items.index.isin(item2id.keys())]
        # map original ids to the internal indices
        items.index = items.index.map(lambda x: item2id[x])

        self.tags = {}
        for tag_index in tags_idxs:
            tags: Tags = item_dtypes[tag_index]
            self.tags[tag_index] = (
                items[tag_index]
                .dropna()
                .str.split(tags.sep, expand=True)
                .stack()
                .unique()
                .tolist()
            )
            items[tag_index] = items[tag_index].fillna("")

        for str_index in string_idxs:
            items[str_index] = items[str_index].fillna("")

        self.items = items

    def __str__(self):
        return f"Dataset '{self.name()}'"
