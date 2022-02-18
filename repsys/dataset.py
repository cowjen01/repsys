# credits: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

import logging
import os
import random
import typing
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type, List, Optional

import numpy as np
import pandas as pd
from bidict import frozenbidict
from pandas import DataFrame
from scipy.sparse import csr_matrix

import repsys.dtypes as dtypes
from repsys.config import read_config
from repsys.dtypes import (
    DataType,
    find_column_by_type,
    filter_columns_by_type
)
from repsys.utils import (
    create_tmp_dir,
    tmp_dir_path,
    remove_tmp_dir,
    unzip_dir,
    zip_dir,
)

logger = logging.getLogger(__name__)


class Split:
    def __init__(self, name: str, interact_matrix: csr_matrix, users_dict: frozenbidict,
                 holdout_matrix: csr_matrix = None) -> None:
        self.name = name
        self.interact_matrix = interact_matrix
        self.holdout_matrix = holdout_matrix
        self.users_dict = users_dict

        if holdout_matrix is None:
            self.complete_matrix = interact_matrix
        else:
            self.complete_matrix = interact_matrix + holdout_matrix


def reindex_data(df: DataFrame, users_map: frozenbidict, items_map: frozenbidict) -> None:
    df['user'] = df['user'].apply(lambda x: users_map[x])
    df['item'] = df['item'].apply(lambda x: items_map[x])


def build_interact_matrix(df: DataFrame, n_items: int) -> csr_matrix:
    # internal indexes start from zero
    n_users = df["user"].max() + 1
    rows, cols, values = df["user"], df["item"], df["value"]

    # put one only at places, where uid and sid index meet
    return csr_matrix(
        (values, (rows, cols)),
        dtype="float64",
        shape=(n_users, n_items),
    )


def build_index_dict(ids) -> frozenbidict:
    return frozenbidict((uid, i) for (i, uid) in enumerate(ids))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_index_dict(file_path: str) -> frozenbidict:
    with open(file_path, "r") as f:
        return frozenbidict({
            line.strip(): i for i, line in enumerate(f)
        })


def save_index_dict(index_dict: frozenbidict, file_path: str) -> None:
    with open(file_path, "w") as f:
        for sid in index_dict.keys():
            f.write("%s\n" % sid)


def csr_matrix_to_df(matrix: csr_matrix) -> DataFrame:
    coo = matrix.tocoo()
    return pd.DataFrame(
        data={"user": coo.row, "item": coo.col, "value": coo.data},
        columns=["user", "item", "value"],
    )


def save_split(split: Split, output_dir: str) -> None:
    interact_data_path = os.path.join(output_dir, f"{split.name}_interact.csv")
    holdout_data_path = os.path.join(output_dir, f"{split.name}_holdout.csv")
    users_dict_path = os.path.join(output_dir, f"{split.name}_users.txt")

    interact_df = csr_matrix_to_df(split.interact_matrix)
    interact_df.to_csv(interact_data_path, index=False)

    if split.holdout_matrix is not None:
        holdout_df = csr_matrix_to_df(split.holdout_matrix)
        holdout_df.to_csv(holdout_data_path, index=False)

    save_index_dict(split.users_dict, users_dict_path)


def load_split(split_name: str, input_dir: str) -> Tuple[
    frozenbidict, DataFrame, Optional[DataFrame]]:
    interact_data_path = os.path.join(input_dir, f"{split_name}_interact.csv")
    holdout_data_path = os.path.join(input_dir, f"{split_name}_holdout.csv")
    users_dict_path = os.path.join(input_dir, f"{split_name}_users.txt")

    interact_df = pd.read_csv(interact_data_path)
    users_dict = load_index_dict(users_dict_path)

    holdout_df = None
    if os.path.isfile(holdout_data_path):
        holdout_df = pd.read_csv(holdout_data_path)

    return users_dict, interact_df, holdout_df


def save_items(items: DataFrame, items_dict: frozenbidict, output_dir: str) -> None:
    items_data_path = os.path.join(output_dir, "items.csv")
    items_dict_path = os.path.join(output_dir, "items.txt")

    items.to_csv(items_data_path, index=True)
    save_index_dict(items_dict, items_dict_path)


def load_items(item_cols: Dict[str, Type[DataType]], input_dir: str) -> Tuple[
    DataFrame, frozenbidict]:
    items_data_path = os.path.join(input_dir, "items.csv")
    items_index_path = os.path.join(input_dir, "items.txt")

    str_dtypes = [dtypes.ItemID, dtypes.Number]
    pandas_dtypes = {col: str for col, dt in item_cols.items() if type(dt) not in str_dtypes}

    items = pd.read_csv(items_data_path, dtype=pandas_dtypes)
    items = items.set_index(find_column_by_type(item_cols, dtypes.ItemID))

    items_dict = load_index_dict(items_index_path)

    return items, items_dict


class Dataset(ABC):
    def __init__(self):
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
    def item_columns(self) -> Dict[str, Type[DataType]]:
        pass

    @abstractmethod
    def interaction_columns(self) -> Dict[str, Type[DataType]]:
        pass

    @abstractmethod
    def load_items(self) -> DataFrame:
        pass

    @abstractmethod
    def load_interactions(self) -> DataFrame:
        pass

    def _get_split(self, split: str = 'train') -> Optional[Split]:
        return self.splits.get(split)

    def _get_title_column(self) -> str:
        item_cols = self.item_columns()
        title_col = find_column_by_type(item_cols, dtypes.Title)
        return title_col

    def _item_id_to_index(self, item_id: int) -> int:
        return self.items_dict.get(item_id)

    def _item_index_to_id(self, item_index: int) -> int:
        return self.items_dict.inverse.get(item_index)

    def _user_id_to_index(self, user_id: int, split: str = 'train') -> int:
        return self.splits.get(split).users_dict.get(user_id)

    def _user_index_to_id(self, user_index: int, split: str = 'train') -> int:
        return self.splits.get(split).users_dict.inverse.get(user_index)

    def _item_indexes_to_ids(self, item_indexes: List[int]) -> List[int]:
        return [self._item_index_to_id(item_index) for item_index in item_indexes]

    def _item_ids_to_indexes(self, item_ids: List[int]) -> List[int]:
        return [self._item_id_to_index(item_id) for item_id in item_ids]

    def get_train_data(self) -> csr_matrix:
        return self.splits.get('train').interact_matrix

    def get_validation_data(self) -> Tuple[csr_matrix, csr_matrix]:
        split = self.splits.get('validation')
        return split.interact_matrix, split.holdout_matrix

    def get_test_data(self) -> Tuple[csr_matrix, csr_matrix]:
        split = self.splits.get('test')
        return split.interact_matrix, split.holdout_matrix

    def get_total_items(self):
        return self.items.shape[0]

    def get_items_by_title(self, query: str) -> DataFrame:
        title_col = self._get_title_column()
        items_filter = self.items[title_col].str.contains(query, case=False)
        return self.items[items_filter]

    def get_interactions_by_user(self, user_id: int, split: str = 'train') -> csr_matrix:
        user_index = self._user_id_to_index(user_id)
        matrix = self._get_split(split).complete_matrix
        return matrix[user_index]

    def get_interacted_items_by_user(self, user_id: int, split: str = 'train') -> DataFrame:
        interactions = self.get_interactions_by_user(user_id, split)
        item_indexes = (interactions > 0).indices
        item_ids = self._item_indexes_to_ids(item_indexes)
        return self.items.loc[item_ids]

    def interactions_to_matrix(self, interactions: List[int]) -> csr_matrix:
        return csr_matrix(
            (np.ones_like(interactions), (np.zeros_like(interactions), interactions)),
            dtype="float64",
            shape=(1, self.get_total_items()),
        )

    def _update_tags(self, items: DataFrame) -> None:
        item_cols = self.item_columns()
        tag_cols = filter_columns_by_type(item_cols, dtypes.Tags)

        for col in tag_cols:
            params = typing.cast(dtypes.Tags, item_cols[col])
            self.tags[col] = items[col].dropna().str.split(params.sep,
                                                           expand=True).stack().unique().tolist()
            items[col] = items[col].fillna("")
            items[col] = items[col].str.split(params.sep)

    def _update_categories(self, items: DataFrame) -> None:
        item_cols = self.item_columns()
        category_cols = filter_columns_by_type(item_cols, dtypes.Category)

        for col in category_cols:
            self.categories[col] = (items[col].unique().tolist())

    def _update_data(self, splits, items: DataFrame, items_dict: frozenbidict) -> None:
        n_items = items.shape[0]

        self.splits['train'] = Split(
            name='train',
            interact_matrix=build_interact_matrix(splits[0][1], n_items),
            users_dict=splits[0][0])
        self.splits['validation'] = Split(
            name='validation',
            interact_matrix=build_interact_matrix(splits[1][1], n_items),
            holdout_matrix=build_interact_matrix(splits[1][2], n_items),
            users_dict=splits[1][0])
        self.splits['test'] = Split(
            name='test',
            interact_matrix=build_interact_matrix(splits[2][1], n_items),
            holdout_matrix=build_interact_matrix(splits[2][2], n_items),
            users_dict=splits[2][0])

        self._update_tags(items)
        self._update_categories(items)

        self.items = items
        self.items_dict = items_dict

    def prepare(self):
        logger.debug("Loading dataset ...")

        items_df = self.load_items()
        item_cols = self.item_columns()
        interactions_df = self.load_interactions()
        interaction_cols = self.interaction_columns()

        logger.debug("Validating dataset ...")

        # validate_dataset(activities_df, items_df, activity_cols, item_cols)

        item_id_col = find_column_by_type(interaction_cols, dtypes.ItemID)
        user_id_col = find_column_by_type(interaction_cols, dtypes.UserID)
        value_col = find_column_by_type(interaction_cols, dtypes.Interaction)

        if not value_col:
            interactions_df['value'] = 1
            value_col = 'value'

        logger.debug("Splitting interactions ...")

        interactions_df = interactions_df[interaction_cols.keys()]
        interactions_df = interactions_df.rename(
            columns={item_id_col: 'item', user_id_col: 'user', value_col: 'value'})

        config = read_config()

        splitter = DatasetSplitter(
            config.dataset.train_split_prop,
            config.dataset.test_holdout_prop,
            config.dataset.user_interactions_threshold,
            config.dataset.item_interactions_threshold,
            config.dataset.interaction_value_threshold,
            config.seed
        )

        train_split, vad_split, test_split = splitter.split(interactions_df)

        train_users_ids, train_data = train_split
        vad_users_ids, vad_train_data, vad_holdout_data = vad_split
        test_users_ids, test_train_data, test_holdout_data = test_split

        items_ids = pd.unique(train_data['item'])

        items_dict = build_index_dict(items_ids)
        train_users_dict = build_index_dict(train_users_ids)
        vad_users_dict = build_index_dict(vad_users_ids)
        test_users_dict = build_index_dict(test_users_ids)

        reindex_data(train_data, train_users_dict, items_dict)

        reindex_data(vad_train_data, vad_users_dict, items_dict)
        reindex_data(vad_holdout_data, vad_users_dict, items_dict)

        reindex_data(test_train_data, test_users_dict, items_dict)
        reindex_data(test_holdout_data, test_users_dict, items_dict)

        # keep only columns defined in the dtypes
        items = items_df[item_cols.keys()]
        # set index column as an index
        items = items.set_index(find_column_by_type(item_cols, dtypes.ItemID))
        # filter only items included in the training data
        items = items[items.index.isin(items_ids)]

        splits = (
            (train_users_dict, train_data),
            (vad_users_dict, vad_train_data, vad_holdout_data),
            (test_users_dict, test_train_data, test_holdout_data),
        )

        self._update_data(splits, items, items_dict)

    def load(self, path: str):
        logger.info(f"Loading dataset from '{path}'")
        create_tmp_dir()
        try:
            unzip_dir(path, tmp_dir_path())
            item_cols = self.item_columns()
            # validate_item_dtypes(item_dtypes)
            items, items_dict = load_items(item_cols, tmp_dir_path())
            # validate_item_data(items, item_dtypes)
            train_split = load_split('train', tmp_dir_path())
            vad_split = load_split('validation', tmp_dir_path())
            test_split = load_split('test', tmp_dir_path())
            self._update_data((train_split, vad_split, test_split), items, items_dict)
        finally:
            remove_tmp_dir()

    def save(self, path: str):
        create_tmp_dir()
        try:
            for split in self.splits.values():
                save_split(split, tmp_dir_path())
            save_items(self.items, self.items_dict, tmp_dir_path())
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
        user_interactions_threshold=5,
        item_interactions_threshold=0,
        interaction_value_threshold=0,
        seed=1234,
        user_col='user',
        item_col='item',
        value_col='value'
    ) -> None:
        self.train_split_prop = train_split_prop
        self.test_holdout_prop = test_holdout_prop
        self.user_interactions_threshold = user_interactions_threshold
        self.item_interactions_threshold = item_interactions_threshold
        self.interaction_value_threshold = interaction_value_threshold
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col
        self.value_col = value_col

    @classmethod
    def get_count(cls, df, col):
        grouped_df = df[[col]].groupby(col, as_index=True)
        count = grouped_df.size()
        return count

    # filter interactions by two conditions (minimal interactions
    # for movie, minimal interactions by user)
    def _filter_triplets(self, tp):
        # Only keep the triplets for items which
        # were clicked on by at least min_sc users.
        if self.item_interactions_threshold > 0:
            item_count = self.get_count(tp, self.item_col)
            tp = tp[
                tp[self.item_col].isin(
                    item_count.index[item_count >= self.item_interactions_threshold])]

        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some items will have less than min_uc users,
        # but should only be a small proportion
        if self.user_interactions_threshold > 0:
            user_count = self.get_count(tp, self.user_col)
            tp = tp[
                tp[self.user_col].isin(
                    user_count.index[user_count >= self.user_interactions_threshold])]

        # Update both user count and item count after filtering
        user_count = self.get_count(tp, self.user_col)
        item_count = self.get_count(tp, self.item_col)

        return tp, user_count, item_count

    def _split_train_test(self, data):
        grouped_by_user = data.groupby(self.user_col)
        tr_list, te_list = list(), list()

        for i, (_, group) in enumerate(grouped_by_user):
            n_items_u = len(group)

            # randomly choose 20% of all items user interacted with
            # these interactions goes to test list, other goes to training list
            idx = np.zeros(n_items_u, dtype="bool")

            set_seed(self.seed)

            idx[np.random.choice(n_items_u,
                                 size=int(self.test_holdout_prop * n_items_u),
                                 replace=False).astype("int64")] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te

    # we will only be working with movies that has been seen by the model, so we need
    # to remove all interactions to movies out of the training scope
    def _filter_interact_data(self, interact_data, users, item_index):
        # filter only interactions made by users
        interactions = interact_data.loc[interact_data[self.user_col].isin(users)]
        # filter only interactions with items included in the item index
        interactions = interactions.loc[interactions[self.item_col].isin(item_index)]
        # filter only interactions meet the main criteria
        # this way we ensure there will be no vad/test user with less
        # than x interactions (this could cause some user gets into the vad-tr set
        # but not into the vad-te set because of not enough interactions)
        interactions, activity, _ = self._filter_triplets(interactions)

        return interactions, activity.index

    def split(self, interact_data) -> Tuple[
        Tuple[List[int], DataFrame], Tuple[List[int], DataFrame, DataFrame], Tuple[
            List[int], DataFrame, DataFrame]]:
        interact_data, user_activity, item_popularity = self._filter_triplets(
            interact_data
        )

        holdout_users_portion = (1 - self.train_split_prop) / 2

        # Shuffle users using permutation
        user_index = user_activity.index

        set_seed(self.seed)

        idx_perm = np.random.permutation(user_index.size)
        # user_index is an array of shuffled users ids
        user_index = user_index[idx_perm]

        n_users = user_index.size
        n_holdout_users = round(n_users * holdout_users_portion)

        # Select 10K users as holdout users, 10K users as validation users
        # and the rest of the users for training
        tr_users = user_index[: (n_users - n_holdout_users * 2)]
        vad_users = user_index[
                    (n_users - n_holdout_users * 2): (n_users - n_holdout_users)]
        test_users = user_index[(n_users - n_holdout_users):]

        # Select only interactions made by users from the training set
        train_interactions = interact_data.loc[
            interact_data[self.user_col].isin(tr_users)]

        # Get all movies interacted by the train users
        # we will only be working with movies that has been seen by model
        item_index = pd.unique(train_interactions[self.item_col])

        # Select only interactions made by the validation users
        # and also those whose movie is included in the training interactions
        vad_interactions, vad_users = self._filter_interact_data(interact_data,
                                                                 vad_users,
                                                                 item_index)
        vad_interactions_tr, vad_interactions_te = self._split_train_test(
            vad_interactions)

        test_interactions, test_users = self._filter_interact_data(interact_data,
                                                                   test_users,
                                                                   item_index)
        test_interactions_tr, test_interactions_te = self._split_train_test(
            test_interactions)

        return (
            (tr_users, train_interactions),
            (vad_users, vad_interactions_tr, vad_interactions_te),
            (test_users, test_interactions_tr, test_interactions_te)
        )
