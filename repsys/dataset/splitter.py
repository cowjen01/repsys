# credits: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from pandas import Index, DataFrame


logger = logging.getLogger(__name__)


class Split:
    def __init__(
        self, train_data: DataFrame, test_data: DataFrame, users: Index
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
            (n_users - n_heldout_users * 2) : (n_users - n_heldout_users)
        ]
        test_users = user_index[(n_users - n_heldout_users) :]

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

        train_split = Split(train_interacts, None, tr_users)
        vad_split = Split(vad_interacts_tr, vad_interacts_te, vad_users)
        test_split = Split(test_interacts_tr, test_interacts_te, test_users)

        return train_split, vad_split, test_split
