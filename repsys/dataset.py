# credits: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

import logging
import os
import random
import typing
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from bidict import frozenbidict
from pandas import DataFrame, Series, Index
from scipy.sparse import csr_matrix

import repsys.dtypes as dtypes
from repsys.config import read_config
from repsys.dtypes import (
    ColumnDict,
    find_column_by_type,
    filter_columns_by_type
)
from repsys.helpers import (
    create_tmp_dir,
    tmp_dir_path,
    remove_tmp_dir,
    unzip_dir,
    zip_dir,
    set_seed
)

logger = logging.getLogger(__name__)


class Split:
    def __init__(self, name: str, train_matrix: csr_matrix, user_index: frozenbidict,
                 holdout_matrix: csr_matrix = None) -> None:
        self.name = name
        self.train_matrix = train_matrix
        self.holdout_matrix = holdout_matrix
        self.user_index = user_index

        if holdout_matrix is None:
            self.complete_matrix = train_matrix
        else:
            self.complete_matrix = train_matrix + holdout_matrix


def reindex_data(df: DataFrame, user_index: frozenbidict, item_index: frozenbidict) -> None:
    df['user'] = df['user'].apply(lambda x: user_index[x])
    df['item'] = df['item'].apply(lambda x: item_index[x])


def df_to_matrix(df: DataFrame, n_items: int) -> csr_matrix:
    n_users = df["user"].max() + 1
    rows, cols, values = df["user"], df["item"], df["value"]

    return csr_matrix(
        (values, (rows, cols)),
        dtype="float64",
        shape=(n_users, n_items),
    )


def matrix_to_df(matrix: csr_matrix) -> DataFrame:
    coo = matrix.tocoo()
    return pd.DataFrame(
        data={"user": coo.row, "item": coo.col, "value": coo.data},
        columns=["user", "item", "value"],
    )


def build_index(ids) -> frozenbidict:
    return frozenbidict((uid, i) for (i, uid) in enumerate(ids))


def load_index(file_path: str) -> frozenbidict:
    with open(file_path, "r") as f:
        return frozenbidict({
            line.strip(): i for i, line in enumerate(f)
        })


def save_index(index_dict: frozenbidict, file_path: str) -> None:
    with open(file_path, "w") as f:
        for sid in index_dict.keys():
            f.write("%s\n" % sid)


def save_split(split: Split, output_dir: str) -> None:
    train_data_path = os.path.join(output_dir, f"{split.name}_train.csv")
    holdout_data_path = os.path.join(output_dir, f"{split.name}_holdout.csv")
    user_index_path = os.path.join(output_dir, f"{split.name}_users.txt")

    train_data = matrix_to_df(split.train_matrix)
    train_data.to_csv(train_data_path, index=False)

    if split.holdout_matrix is not None:
        holdout_data = matrix_to_df(split.holdout_matrix)
        holdout_data.to_csv(holdout_data_path, index=False)

    save_index(split.user_index, user_index_path)


def load_split(split_name: str, input_dir: str) -> Tuple[frozenbidict, DataFrame, Optional[DataFrame]]:
    train_data_path = os.path.join(input_dir, f"{split_name}_train.csv")
    holdout_data_path = os.path.join(input_dir, f"{split_name}_holdout.csv")
    user_index_path = os.path.join(input_dir, f"{split_name}_users.txt")

    train_data = pd.read_csv(train_data_path)
    user_index = load_index(user_index_path)

    holdout_data = None
    if os.path.isfile(holdout_data_path):
        holdout_data = pd.read_csv(holdout_data_path)

    return user_index, train_data, holdout_data


def save_items(items: DataFrame, columns: ColumnDict, item_index: frozenbidict,
               output_dir: str) -> None:
    data_path = os.path.join(output_dir, "items.csv")
    index_path = os.path.join(output_dir, "items.txt")

    save_copy = items.copy()

    tag_cols = filter_columns_by_type(columns, dtypes.Tag)
    for col in tag_cols:
        save_copy[col] = save_copy[col].str.join(";")

    save_copy.to_csv(data_path, index=True)
    save_index(item_index, index_path)


def load_items(item_cols: ColumnDict, input_dir: str) -> Tuple[DataFrame, frozenbidict]:
    data_path = os.path.join(input_dir, "items.csv")
    index_path = os.path.join(input_dir, "items.txt")

    items_dtypes = {col: str for col, dt in item_cols.items() if type(dt) != dtypes.Number}

    items = pd.read_csv(data_path, dtype=items_dtypes)
    items = items.set_index(find_column_by_type(item_cols, dtypes.ItemID))

    tag_cols = filter_columns_by_type(item_cols, dtypes.Tag)
    for col in tag_cols:
        items[col] = items[col].str.split(';')

    item_index = load_index(index_path)

    return items, item_index


class Dataset(ABC):
    def __init__(self):
        self.items: Optional[DataFrame] = None
        self.item_index: Optional[frozenbidict] = None
        self.tags = {}
        self.categories = {}
        self.splits: Dict[str, Optional[Split]] = {
            'train': None,
            'validation': None,
            'test': None
        }

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def item_cols(self) -> ColumnDict:
        pass

    @abstractmethod
    def interaction_cols(self) -> ColumnDict:
        pass

    @abstractmethod
    def load_items(self) -> DataFrame:
        pass

    @abstractmethod
    def load_interactions(self) -> DataFrame:
        pass

    def get_split(self, split: str = 'train') -> Optional[Split]:
        return self.splits.get(split)

    def _get_title_col(self) -> str:
        return find_column_by_type(self.item_cols(), dtypes.Title)

    def iid_to_index(self, iid: str) -> int:
        return self.item_index.get(iid)

    def index_to_iid(self, index: int) -> int:
        return self.item_index.inverse.get(index)

    def uid_to_index(self, uid: int, split: str = 'train') -> int:
        return self.splits.get(split).user_index.get(uid)

    def index_to_uid(self, index: int, split: str = 'train') -> int:
        return self.splits.get(split).user_index.inverse.get(index)

    def get_train_data(self) -> csr_matrix:
        return self.splits.get('train').train_matrix

    def get_vad_data(self) -> Tuple[csr_matrix, csr_matrix]:
        split = self.splits.get('validation')
        return split.train_matrix, split.holdout_matrix

    def get_test_data(self) -> Tuple[csr_matrix, csr_matrix]:
        split = self.splits.get('test')
        return split.train_matrix, split.holdout_matrix

    def get_total_items(self):
        return self.items.shape[0]

    def get_items_by_title(self, query: str) -> DataFrame:
        col = self._get_title_col()
        item_filter = self.items[col].str.contains(query, case=False)
        return self.items[item_filter]

    def get_interactions_by_user(self, uid: int, split: str = 'train') -> csr_matrix:
        index = self.index_to_uid(uid, split)
        matrix = self.get_split(split).complete_matrix
        return matrix[index]

    def get_interacted_items_by_user(self, user_id: int, split: str = 'train') -> DataFrame:
        interactions = self.get_interactions_by_user(user_id, split)
        indexes = (interactions > 0).indices
        ids = list(map(self.index_to_iid, indexes))
        return self.items.loc[ids]

    def interactions_to_matrix(self, interactions: List[int]) -> csr_matrix:
        return csr_matrix(
            (np.ones_like(interactions), (np.zeros_like(interactions), interactions)),
            dtype="float64",
            shape=(1, self.get_total_items()),
        )

    def _update_tags(self, items: DataFrame) -> None:
        cols = filter_columns_by_type(self.item_cols(), dtypes.Tag)
        for col in cols:
            self.tags[col] = np.unique(np.concatenate(items[col].values)).tolist()

    def _update_categories(self, items: DataFrame) -> None:
        cols = filter_columns_by_type(self.item_cols(), dtypes.Category)
        for col in cols:
            self.categories[col] = items[col].unique().tolist()

    def _update_data(self, splits, items: DataFrame, item_index: frozenbidict) -> None:
        n_items = items.shape[0]

        self.splits['train'] = Split(
            name='train',
            train_matrix=df_to_matrix(splits[0][1], n_items),
            user_index=splits[0][0])
        self.splits['validation'] = Split(
            name='validation',
            train_matrix=df_to_matrix(splits[1][1], n_items),
            holdout_matrix=df_to_matrix(splits[1][2], n_items),
            user_index=splits[1][0])
        self.splits['test'] = Split(
            name='test',
            train_matrix=df_to_matrix(splits[2][1], n_items),
            holdout_matrix=df_to_matrix(splits[2][2], n_items),
            user_index=splits[2][0])

        self._update_tags(items)
        self._update_categories(items)

        self.items = items
        self.item_index = item_index

    def prepare(self) -> None:
        logger.debug("Loading dataset ...")

        items = self.load_items()
        item_cols = self.item_cols()
        interacts = self.load_interactions()
        interaction_cols = self.interaction_cols()

        logger.debug("Validating dataset ...")

        # validate_dataset(activities_df, items, activity_cols, item_cols)

        interacts_item_col = find_column_by_type(interaction_cols, dtypes.ItemID)
        interacts_user_col = find_column_by_type(interaction_cols, dtypes.UserID)
        interacts_value_col = find_column_by_type(interaction_cols, dtypes.Interaction)

        interacts[interacts_item_col] = interacts[interacts_item_col].astype(str)
        interacts[interacts_user_col] = interacts[interacts_user_col].astype(str)

        if not interacts_item_col:
            interacts['value'] = 1
            interacts_value_col = 'value'

        logger.debug("Splitting interactions ...")

        interactions = interacts[interaction_cols.keys()]
        interactions = interactions.rename(
            columns={interacts_item_col: 'item', interacts_user_col: 'user', interacts_value_col: 'value'})

        config = read_config()

        splitter = DatasetSplitter(
            config.dataset.train_split_prop,
            config.dataset.test_holdout_prop,
            config.dataset.min_user_interacts,
            config.dataset.min_item_interacts,
            config.dataset.min_interact_value,
            config.seed
        )

        train_split, vad_split, test_split = splitter.split(interactions)

        train_user_ids, train_data = train_split
        vad_user_ids, vad_train_data, vad_holdout_data = vad_split
        test_user_ids, test_train_data, test_holdout_data = test_split

        item_ids = pd.unique(train_data['item'])

        item_index = build_index(item_ids)
        train_user_index = build_index(train_user_ids)
        vad_user_index = build_index(vad_user_ids)
        test_user_index = build_index(test_user_ids)

        reindex_data(train_data, train_user_index, item_index)

        reindex_data(vad_train_data, vad_user_index, item_index)
        reindex_data(vad_holdout_data, vad_user_index, item_index)

        reindex_data(test_train_data, test_user_index, item_index)
        reindex_data(test_holdout_data, test_user_index, item_index)

        # keep only columns defined in the dtypes
        items = items[item_cols.keys()]

        items_id_col = find_column_by_type(item_cols, dtypes.ItemID)
        items[items_id_col] = items[items_id_col].astype(str)
        items = items.set_index(items_id_col)

        # filter only items included in the training data
        items = items[items.index.isin(item_ids)]

        tag_cols = filter_columns_by_type(item_cols, dtypes.Tag)
        for col in tag_cols:
            params = typing.cast(dtypes.Tag, item_cols[col])
            items[col] = items[col].fillna("")
            items[col] = items[col].str.split(params.sep)

        splits = (
            (train_user_index, train_data),
            (vad_user_index, vad_train_data, vad_holdout_data),
            (test_user_index, test_train_data, test_holdout_data),
        )

        self._update_data(splits, items, item_index)

    def load(self, path: str) -> None:
        logger.info(f"Loading dataset from '{path}'")
        create_tmp_dir()
        try:
            unzip_dir(path, tmp_dir_path())
            # validate_item_dtypes(item_dtypes)
            items, item_index = load_items(self.item_cols(), tmp_dir_path())
            # validate_item_data(items, item_dtypes)
            train_split = load_split('train', tmp_dir_path())
            vad_split = load_split('validation', tmp_dir_path())
            test_split = load_split('test', tmp_dir_path())

            splits = train_split, vad_split, test_split

            self._update_data(splits, items, item_index)
        finally:
            remove_tmp_dir()

    def save(self, path: str) -> None:
        create_tmp_dir()
        try:
            for split in self.splits.values():
                save_split(split, tmp_dir_path())

            save_items(self.items, self.item_cols(), self.item_index, tmp_dir_path())
            zip_dir(path, tmp_dir_path())
        finally:
            remove_tmp_dir()

    def __str__(self):
        return f"Dataset '{self.name()}'"


class DatasetSplitter:
    def __init__(
        self,
        train_split_prop=0.85,
        test_holdout_prop=0.2,
        min_user_interacts=5,
        min_item_interacts=0,
        min_interact_value=0,
        seed=1234,
        user_col='user',
        item_col='item',
        value_col='value'
    ) -> None:
        self.train_split_prop = train_split_prop
        self.test_holdout_prop = test_holdout_prop
        self.min_user_interacts = min_user_interacts
        self.min_item_interacts = min_item_interacts
        self.min_interact_value = min_interact_value
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col
        self.value_col = value_col

    @classmethod
    def get_count(cls, df: DataFrame, col: str) -> Series:
        grouped_df = df[[col]].groupby(col, as_index=True)
        count = grouped_df.size()
        return count

    # filter interactions by two conditions (minimal interactions
    # for movie, minimal interactions by user)
    def _filter_triplets(self, df: DataFrame) -> Tuple[DataFrame, Series, Series]:
        if self.min_interact_value > 0:
            df = df[df[self.value_col] >= self.min_interact_value]

        # Only keep the triplets for items which
        # were clicked on by at least min_sc users.
        if self.min_item_interacts > 0:
            item_popularity = self.get_count(df, self.item_col)
            item_filter = item_popularity.index[item_popularity >= self.min_item_interacts]
            df = df[df[self.item_col].isin(item_filter)]

        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some items will have less than min_uc users,
        # but should only be a small proportion
        if self.min_user_interacts > 0:
            user_activity = self.get_count(df, self.user_col)
            user_filter = user_activity.index[user_activity >= self.min_user_interacts]
            df = df[df[self.user_col].isin(user_filter)]

        # Update both user count and item count after filtering
        user_activity = self.get_count(df, self.user_col)
        item_popularity = self.get_count(df, self.item_col)

        return df, user_activity, item_popularity

    def _split_holdout(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        grouped_by_user = df.groupby(self.user_col)
        train_list, holdout_list = list(), list()

        for i, (_, group) in enumerate(grouped_by_user):
            n_items = len(group)

            # randomly choose 20% of all items user interacted with
            # these interactions goes to test list, other goes to training list
            indexes = np.zeros(n_items, dtype="bool")
            holdout_size = int(self.test_holdout_prop * n_items)

            set_seed(self.seed)
            rand_items = np.random.choice(n_items, size=holdout_size, replace=False).astype("int64")
            indexes[rand_items] = True

            train_list.append(group[np.logical_not(indexes)])
            holdout_list.append(group[indexes])

        train_data = pd.concat(train_list)
        holdout_data = pd.concat(holdout_list)

        return train_data, holdout_data

    # we will only be working with movies that has been seen by the model, so we need
    # to remove all interactions to movies out of the training scope
    def _filter_interact_data(self, df: DataFrame, user_index: Index, item_index: Index) -> Tuple[DataFrame, Index]:
        # filter only interactions made by users
        df = df.loc[df[self.user_col].isin(user_index)]
        # filter only interactions with items included in the item index
        df = df.loc[df[self.item_col].isin(item_index)]
        # filter only interactions meet the main criteria
        # this way we ensure there will be no vad/test user with less
        # than x interactions (this could cause some user gets into the vad-tr set
        # but not into the vad-te set because of not enough interactions)
        df, user_activity, _ = self._filter_triplets(df)

        return df, user_activity.index

    def split(self, df: DataFrame) -> Tuple[
        Tuple[Index, DataFrame], Tuple[Index, DataFrame, DataFrame], Tuple[Index, DataFrame, DataFrame]]:

        df, user_activity, item_popularity = self._filter_triplets(df)
        user_index = user_activity.index

        # shuffle users using permutation
        set_seed(self.seed)
        index_perm = np.random.permutation(user_index.size)
        # user_index is an array of shuffled users ids
        user_index = user_index[index_perm]

        n_users = user_index.size
        n_holdout_users = round(n_users * (1 - self.train_split_prop) / 2)

        # select 10K users as holdout users, 10K users as validation users
        # and the rest of the users for training
        train_users = user_index[: (n_users - n_holdout_users * 2)]
        vad_users = user_index[(n_users - n_holdout_users * 2): (n_users - n_holdout_users)]
        test_users = user_index[(n_users - n_holdout_users):]

        # select only interactions made by users from the training set
        train_data = df.loc[df[self.user_col].isin(train_users)]
        item_index = pd.unique(train_data[self.item_col])

        # select only interactions made by the validation users
        # and also those whose movie is included in the training interactions
        vad_data, vad_users = self._filter_interact_data(df, vad_users, item_index)
        vad_train_data, vad_holdout_data = self._split_holdout(vad_data)

        test_data, test_users = self._filter_interact_data(df, test_users, item_index)
        test_train_data, test_holdout_data = self._split_holdout(test_data)

        return (
            (train_users, train_data),
            (vad_users, vad_train_data, vad_holdout_data),
            (test_users, test_train_data, test_holdout_data)
        )
