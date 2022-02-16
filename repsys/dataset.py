# credits: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Text, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import sparse

from repsys.dtypes import (
    DataType,
    ItemID,
    Number,
    Tags,
    Title,
    UserID,
    Rating,
    find_column,
    filter_columns,
)
from repsys.utils import (
    create_tmp_dir,
    tmp_dir_path,
    remove_tmp_dir,
    unzip_dir,
    zip_dir,
)
from repsys.validators import (
    validate_dataset,
    validate_item_data,
    validate_item_dtypes,
)

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

    def get_user_id(self, user_idx: int) -> int:
        return self._idx2user.get(user_idx)

    def get_item_id(self, item_idx: int) -> int:
        return self._idx2item.get(item_idx)

    def get_user_index(self, user_id: int) -> int:
        return self._user2idx.get(user_id)

    def get_item_index(self, item_id: int) -> int:
        return self._item2idx.get(item_id)

    def filter_items(self, column, query):
        items_filter = self.items[column].str.contains(query, case=False)
        return self.items[items_filter]

    def item_title_col(self):
        dtypes = self.item_dtypes()
        title_col = find_column(dtypes, Title)
        return title_col

    def get_item_ids(self, item_idxs):
        return np.array([self.get_item_id(idx) for idx in item_idxs])

    def get_user_history(self, user_id):
        user_idx = self.get_user_index(user_id)
        return self.vad_data_tr[user_idx]

    def get_interacted_items(self, user_id):
        interactions = self.get_user_history(user_id)
        item_idxs = (interactions > 0).indices
        item_ids = self.get_item_ids(item_idxs)
        return self.items.loc[item_ids]

    def indices_to_items(self, item_idxs):
        item_ids = self.get_item_ids(item_idxs)
        return self.items.loc[item_ids]

    def serialize_items(self, items: DataFrame):
        serialized_items = items.copy()
        serialized_items["id"] = serialized_items.index
        return serialized_items.to_dict("records")

    def input_from_interactions(self, interactions):
        return sparse.csr_matrix(
            (
                np.ones_like(interactions),
                (np.zeros_like(interactions), interactions),
            ),
            dtype="float64",
            shape=(1, self.n_items),
        )

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

        # we need to get a bottom and top index of the vad/test users
        start_idx = min(tp_tr["user_idx"].min(), tp_te["user_idx"].min())
        end_idx = max(tp_tr["user_idx"].max(), tp_te["user_idx"].max())

        # shift all indexes by a number of the train users (in a case of the validation users)
        # or the validation users (in a case of test users) to start from a zero
        rows_tr, cols_tr = tp_tr["user_idx"] - start_idx, tp_tr["item_idx"]
        rows_te, cols_te = tp_te["user_idx"] - start_idx, tp_te["item_idx"]

        # the first vad/test user is at the zero position of the sparse matrix
        # to get his ID, we need to take the index (e.g. 0) and add the total
        # number of the train users to get the index in the dictionary (user_index.txt)
        # at the line number xyz will be user's ID
        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float64",
            # a size of the sparse matrix is a difference between the lowest index
            # and the highest index of the current vad/test set
            shape=(end_idx - start_idx + 1, n_items),
        )

        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, n_items),
        )

        return data_tr, data_te

    def _update_data(
        self,
        splits,
        items: DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
    ):
        item_dtypes = self.item_dtypes()

        self._raw_train_data = splits[0]
        self._raw_vad_data = splits[1]
        self._raw_test_data = splits[2]

        self._user2idx = user2idx
        self._item2idx = item2idx

        self._idx2user = {y: x for x, y in user2idx.items()}
        self._idx2item = {y: x for x, y in item2idx.items()}

        self.train_data, self.n_items = self._build_train_matrix(
            self._raw_train_data
        )
        self.vad_data_tr, self.vad_data_te = self._build_tr_te_matrix(
            self._raw_vad_data, self.n_items
        )
        self.test_data_tr, self.test_data_te = self._build_tr_te_matrix(
            self._raw_test_data, self.n_items
        )

        vad_users_idxs = self._raw_vad_data[0]['user_idx'].unique()
        self.vad_users = list(map(lambda x: self._idx2user[x], vad_users_idxs))

        logger.debug("Processing items data ...")

        tags_columns = filter_columns(item_dtypes, Tags)

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
            items[tags_col] = items[tags_col].str.split(tags.sep)

        self.items = items
        self._is_fitted = True

    def fit(self):
        logger.debug("Loading dataset ...")

        items = self.load_items()
        item_dtypes = self.item_dtypes()
        interacts = self.load_interacts()
        interact_dtypes = self.interact_dtypes()

        logger.debug("Validating dataset ...")

        validate_dataset(interacts, items, interact_dtypes, item_dtypes)

        item_id_col = find_column(interact_dtypes, ItemID)
        user_id_col = find_column(interact_dtypes, UserID)
        rating_col = find_column(interact_dtypes, Rating)

        if rating_col:
            rating: Rating = interact_dtypes[rating_col]
            interacts = interacts[interacts[rating_col] >= rating.bin_threshold]

        logger.debug("Splitting interactions ...")

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
        logger.info(f"Loading dataset from '{path}'")

        create_tmp_dir()

        try:
            unzip_dir(path, tmp_dir_path())

            item_dtypes = self.item_dtypes()
            validate_item_dtypes(item_dtypes)

            str_dtypes = {
                col: str
                for col, dt in item_dtypes.items()
                if type(dt) != ItemID and type(dt) != Number
            }

            logger.debug("Loading items data ...")
            items_data_path = os.path.join(tmp_dir_path(), "items.csv")
            items = pd.read_csv(items_data_path, dtype=str_dtypes)

            logger.debug("Validating items ...")

            validate_item_data(items, item_dtypes)

            logger.debug("Loading interaction data ...")

            train_data_path = os.path.join(tmp_dir_path(), "train.csv")
            train_data = pd.read_csv(train_data_path)

            vad_data = self._load_tr_te_data("vad")
            test_data = self._load_tr_te_data("test")

            item2ids = self._load_idx_data("item_index.txt")
            user2idx = self._load_idx_data("user_index.txt")

            splits = train_data, vad_data, test_data

            items = items.set_index(find_column(item_dtypes, ItemID))

            self._update_data(splits, items, user2idx, item2ids)

        finally:
            remove_tmp_dir()

    @enforce_fitted
    def save(self, path: Text):
        create_tmp_dir()

        try:
            train_data_path = os.path.join(tmp_dir_path(), "train.csv")
            self._raw_train_data.to_csv(train_data_path, index=False)

            items_data_path = os.path.join(tmp_dir_path(), "items.csv")
            self.items.to_csv(items_data_path, index=True)

            self._save_tr_te_data(self._raw_vad_data, "vad")
            self._save_tr_te_data(self._raw_test_data, "test")

            self._save_idx_data(self._item2idx, "item_index.txt")
            self._save_idx_data(self._user2idx, "user_index.txt")

            zip_dir(path, tmp_dir_path())

        finally:
            remove_tmp_dir()

    def __str__(self):
        return f"Dataset '{self.name()}'"


class Split:
    def __init__(
        self, train_data: DataFrame, test_data: DataFrame, users
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.users = users


class DatasetSplitter:
    def __init__(
        self,
        user_col,
        item_col,
        train_users_prop=0.85,
        test_holdout_prop=0.2,
        min_user_interacts=5,
        min_item_interacts=0,
    ) -> None:
        self.user_col = user_col
        self.item_col = item_col
        self.train_users_prop = train_users_prop
        self.test_holdout_prop = test_holdout_prop
        self.min_user_interacts = min_user_interacts
        self.min_item_interacts = min_item_interacts

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
        if self.min_item_interacts > 0:
            itemcount = self.get_count(tp, self.item_col)
            tp = tp[
                tp[self.item_col].isin(
                    itemcount.index[itemcount >= self.min_item_interacts]
                )
            ]

        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users,
        # but should only be a small proportion
        if self.min_user_interacts > 0:
            usercount = self.get_count(tp, self.user_col)
            tp = tp[
                tp[self.user_col].isin(
                    usercount.index[usercount >= self.min_user_interacts]
                )
            ]

        # Update both usercount and itemcount after filtering
        usercount, itemcount = self.get_count(
            tp, self.user_col
        ), self.get_count(tp, self.item_col)
        return tp, usercount, itemcount

    def _split_train_test(self, data):
        grouped_by_user = data.groupby(self.user_col)
        tr_list, te_list = list(), list()

        np.random.seed(98765)

        for i, (_, group) in enumerate(grouped_by_user):
            n_items_u = len(group)

            # randomly choose 20% of all items user interacted with
            # these interactions goes to test list, other goes to training list
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(
                    n_items_u,
                    size=int(self.test_holdout_prop * n_items_u),
                    replace=False,
                ).astype("int64")
            ] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te

    # we will only be working with movies that has been seen by the model
    # so we need to remove all interactions to movies out of the training scope
    def _filter_interact_data(self, interact_data, users, item_index):
        # filter only interactions made by users
        interacts = interact_data.loc[interact_data[self.user_col].isin(users)]
        # filter only interactions with items included in the item index
        interacts = interacts.loc[interacts[self.item_col].isin(item_index)]
        # filter only interactions meet the main criteria
        # this way we ensure there will be no vad/test user with less
        # than x interactions (this could cause some user gets into the vad-tr set
        # but not into the vad-te set because of not enough interactions)
        interacts, activity, _ = self._filter_triplets(interacts)

        return interacts, activity.index

    def split(self, interact_data) -> Tuple[Split, Split, Split]:
        interact_data, user_activity, item_popularity = self._filter_triplets(
            interact_data
        )

        # sparsity = (
        #     1.0
        #     * interact_data.shape[0]
        #     / (user_activity.shape[0] * item_popularity.shape[0])
        # )

        # logger.debug(
        #     "After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)"
        #     % (
        #         interact_data.shape[0],
        #         user_activity.shape[0],
        #         item_popularity.shape[0],
        #         sparsity * 100,
        #     )
        # )

        heldout_users_portion = (1 - self.train_users_prop) / 2

        # Shuffle users using permutation
        user_index = user_activity.index

        np.random.seed(98765)

        idx_perm = np.random.permutation(user_index.size)
        # user_index is an array of shuffled users ids
        user_index = user_index[idx_perm]

        n_users = user_index.size
        n_heldout_users = round(n_users * heldout_users_portion)

        # Select 10K users as heldout users, 10K users as validation users
        # and the rest of the users for training
        tr_users = user_index[: (n_users - n_heldout_users * 2)]
        vad_users = user_index[
                    (n_users - n_heldout_users * 2): (n_users - n_heldout_users)
                    ]
        test_users = user_index[(n_users - n_heldout_users):]

        # Select only interactions made by users from the training set
        train_interacts = interact_data.loc[
            interact_data[self.user_col].isin(tr_users)
        ]

        # Get all movies interacted by the train users
        # we will only be working with movies that has been seen by model
        item_index = pd.unique(train_interacts[self.item_col])

        # Select only interactions made by the validation users
        # and also those whose movie is included in the training interactions
        vad_interacts, vad_users = self._filter_interact_data(
            interact_data, vad_users, item_index
        )
        vad_interacts_tr, vad_interacts_te = self._split_train_test(
            vad_interacts
        )

        test_interacts, test_users = self._filter_interact_data(
            interact_data, test_users, item_index
        )
        test_interacts_tr, test_interacts_te = self._split_train_test(
            test_interacts
        )

        # train users come from the original shuffled index
        # but the vad/test users are fetched from the dataframe index
        # which is created during adhoc triplets filtration process
        # this is a reason, why the users are sorted by their ID
        train_split = Split(train_interacts, None, tr_users)
        vad_split = Split(vad_interacts_tr, vad_interacts_te, vad_users)
        test_split = Split(test_interacts_tr, test_interacts_te, test_users)

        return train_split, vad_split, test_split
