import glob
import os
import pandas as pd
import shutil

from repsys.utils import remove_dir, create_dir


class DatasetPersistor:
    def __init__(self, dirname=".repsys_checkpoints") -> None:
        self.dir_path = os.path.join(os.getcwd(), dirname)

    def _datapoints_dir_path(self):
        return os.path.join(self.dir_path, "datasets")

    def _tmp_dir_path(self):
        return os.path.join(self.dir_path, "tmp")

    def _datapoints_zip_files(self):
        data_path = self._datapoints_dir_path()
        return glob.glob(os.path.join(data_path, "*.zip"))

    def datapoints_exist(self):
        zip_files = self._datapoints_zip_files()
        return len(zip_files) > 0

    def _unzip_latest_datapoint(self):
        zip_files = self._datapoints_zip_files()
        zip_files.sort()

        create_dir(self._tmp_dir_path())

        shutil.unpack_archive(zip_files[0], self._tmp_dir_path())

    def _load_tr_te_data(self, split_type):
        tr_data_path = os.path.join(
            self._tmp_dir_path(), f"{split_type}_tr.csv"
        )
        te_data_path = os.path.join(
            self._tmp_dir_path(), f"{split_type}_te.csv"
        )

        tp_tr = pd.read_csv(tr_data_path)
        tp_te = pd.read_csv(te_data_path)

        return tp_tr, tp_te

    def _load_items_data(self):
        items_data_path = os.path.join(self._tmp_dir_path(), "items.csv")
        items = pd.read_csv(items_data_path, index_col="index")

        items["id"] = items.index
        # TODO: this put away!
        items["subtitle"] = items["subtitle"].fillna("")
        items = items.fillna("")

        return items

    def _load_vad_users_data(self):
        users_data_path = os.path.join(self._tmp_dir_path(), "users.csv")
        users = pd.read_csv(users_data_path, index_col="index")

        users["label"] = users["label"].astype(str)
        users["id"] = users.index

        return users

    def _load_train_data(self):
        train_data_path = os.path.join(self._tmp_dir_path(), "train.csv")
        train_data = pd.read_csv(train_data_path)

        return train_data

    def load(self):
        self._unzip_latest_datapoint()

        train_data = self._load_train_data()

        items = self._load_items_data()
        vad_users = self._load_vad_users_data()

        vad_data_tr, vad_data_te = self._load_tr_te_data("validation")
        test_data_tr, test_data_te = self._load_tr_te_data("test")

        remove_dir(self._tmp_dir_path())

        return (
            items,
            vad_users,
            train_data,
            (vad_data_tr, vad_data_te),
            (test_data_tr, test_data_te),
        )
