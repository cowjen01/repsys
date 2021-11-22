import logging
import zipfile
import glob
import os
import pandas as pd
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class Dataset:
    def name(self):
        raise NotImplementedError("You must implement the `name` method")

    def _processed_data_path(self):
        return os.path.join(os.getcwd(), "datasets")

    def _processed_data_files(self):
        data_path = self._processed_data_path()

        return glob.glob(os.path.join(data_path, "*.zip"))

    def _proccessed_data_exist(self):
        data_path = self._processed_data_path()

        if not os.path.exists(data_path):
            return False

        data_files = self._processed_data_files()

        return len(data_files) > 0

    def _unzipped_data_path(self):
        return os.path.join(os.getcwd(), "tmp")

    def _unzip_processed_data(self):
        data_files = self._processed_data_files()
        data_files.sort()

        tmp_dir_path = self._unzipped_data_path()
        if not os.path.exists(tmp_dir_path):
            os.makedirs(tmp_dir_path)

        with zipfile.ZipFile(data_files[0], "r") as zip_ref:
            zip_ref.extractall(tmp_dir_path)

    def _load_train_data(self, n_items):
        train_data_path = os.path.join(self._unzipped_data_path(), "train.csv")

        tp = pd.read_csv(train_data_path)

        # internal indexes start from zero
        n_users = tp["uid"].max() + 1
        rows, cols = tp["uid"], tp["sid"]

        # put one only at places, where uid and sid index meet
        data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype="float64",
            shape=(n_users, n_items),
        )

        return data

    def _load_tr_te_data(self, data_type, n_items):
        tr_data_path = os.path.join(
            self._unzipped_data_path(), f"{data_type}_tr.csv"
        )

        te_data_path = os.path.join(
            self._unzipped_data_path(), f"{data_type}_te.csv"
        )

        tp_tr = pd.read_csv(tr_data_path)
        tp_te = pd.read_csv(te_data_path)

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

    def _load_items_data(self):
        items_data_path = os.path.join(self._unzipped_data_path(), "items.csv")

        items = pd.read_csv(items_data_path, index_col="index")
        items["id"] = items.index
        items["subtitle"] = items["subtitle"].fillna('')
        items = items.fillna("")

        return items

    def _load_vad_users_data(self):
        users_data_path = os.path.join(self._unzipped_data_path(), "users.csv")

        users = pd.read_csv(users_data_path, index_col="index")
        users["label"] = users["label"].astype(str)
        users["id"] = users.index

        return users

    def load_dataset(self):
        logger.info(f"Loading '{self.name()}' dataset ...")

        logger.debug("Checking processed data presence ...")
        if os.path.exists(self._unzipped_data_path()):
            logger.debug("Processed data found, loading ...")
        elif self._proccessed_data_exist():
            logger.debug("Compressed data found, unzipping ...")
            self._unzip_processed_data()
        else:
            logger.info("No proccessed data found, processing now ...")

        self.items = self._load_items_data()
        self.n_items = self.items.shape[0]

        self.train_data = self._load_train_data(self.n_items)
        self.n_users = self.train_data.shape[0]

        self.vad_users = self._load_vad_users_data()

        logger.info(f"Loaded {self.n_items} items and {self.n_users} users.")

        self.vad_data_tr, self.vad_data_te = self._load_tr_te_data(
            data_type="validation",
            n_items=self.n_items,
        )
        self.test_data_tr, self.test_data_te = self._load_tr_te_data(
            data_type="test", n_items=self.n_items
        )

        logger.debug(f"Dataset '{self.name()}' successfully loaded.")

    def __str__(self):
        return f"Dataset '{self.name()}'"
