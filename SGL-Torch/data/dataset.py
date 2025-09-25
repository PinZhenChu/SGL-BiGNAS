import scipy.sparse as sp
import os
import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict
from reckit import typeassert, pad_sequences

_USER = "user"
_ITEM = "item"
_RATING = "rating"
_TIME = "time"
_column_dict = {
    "UI": [_USER, _ITEM],
    "UIR": [_USER, _ITEM, _RATING],
    "UIT": [_USER, _ITEM, _TIME],
    "UIRT": [_USER, _ITEM, _RATING, _TIME]
}


class Interaction(object):
    @typeassert(data=(pd.DataFrame, None), num_users=(int, None), num_items=(int, None))
    def __init__(self, data=None, num_users=None, num_items=None):
        if data is None or data.empty:
            self._data = pd.DataFrame()
            self.num_users = 0
            self.num_items = 0
            self.num_ratings = 0
        else:
            self._data = data
            self.num_users = num_users if num_users is not None else max(data[_USER]) + 1
            self.num_items = num_items if num_items is not None else max(data[_ITEM]) + 1
            self.num_ratings = len(data)

    def to_user_item_pairs(self):
        """Return numpy array of (user, item) pairs."""
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        ui_pairs = self._data[[_USER, _ITEM]].to_numpy(copy=True, dtype=np.int32)
        return ui_pairs

    def to_csr_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        users, items = self._data[_USER].to_numpy(), self._data[_ITEM].to_numpy()
        ratings = np.ones(len(users), dtype=np.float32)
        csr_mat = sp.csr_matrix((ratings, (users, items)), shape=(self.num_users, self.num_items))
        return csr_mat

    def to_user_dict(self, by_time=False):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return {}
        user_dict = OrderedDict()
        user_grouped = self._data.groupby(_USER)
        for user, user_data in user_grouped:
            if by_time and _TIME in user_data:
                user_data = user_data.sort_values(by=[_TIME])
            user_dict[user] = user_data[_ITEM].to_numpy(dtype=np.int32)
        return user_dict

    def __len__(self):
        return len(self._data)


class Dataset(object):
    def __init__(self, data_dir, dataset_name, sep, columns, config=None):
        self.data_name = dataset_name
        if self.data_name == 'all_data':
            self.source_domain = None
            self.target_domain = None
        else:
            self.source_domain = config["source_domain"]
            self.target_domain = config["target_domain"]
        # self.source_domain = config["source_domain"]
        # self.target_domain = config["target_domain"]

        # metadata
        self.train_data = Interaction()
        self.valid_data = Interaction()
        self.test_data = Interaction()

        # statistic
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0

        self._load_data(data_dir, sep, columns)
        self.train_csr_mat = self.train_data.to_csr_matrix()
        print("Data loading finished")

    def _load_data(self, data_dir, sep, columns):
        if columns not in _column_dict:
            raise ValueError(f"'columns' must be one of {list(_column_dict.keys())}")
        columns = _column_dict[columns]

        # === 若 dataset_name == 'all_data'，只讀 all_data.train ===
        if getattr(self, 'data_name', None) == 'all_data':
            all_data_file = os.path.join(data_dir, 'all_data.train')
            if not os.path.isfile(all_data_file):
                raise FileNotFoundError(f"{all_data_file} not found!")
            _train_data = pd.read_csv(all_data_file, sep=sep, header=None, names=columns)
            print(f"[INFO] Loaded {all_data_file}, {len(_train_data)} interactions")
            _valid_data = pd.DataFrame()
            _test_data = pd.DataFrame()
            all_data = _train_data
            self.num_users = max(all_data[_USER]) + 1
            self.num_items = max(all_data[_ITEM]) + 1
            self.num_ratings = len(all_data)
            self.num_train_ratings = len(_train_data)
            self.train_data = Interaction(_train_data, num_users=self.num_users, num_items=self.num_items)
            self.valid_data = Interaction(_valid_data, num_users=self.num_users, num_items=self.num_items)
            self.test_data = Interaction(_test_data, num_users=self.num_users, num_items=self.num_items)
            self.num_source_items = 0
            self.num_target_items = 0
            print(f"[INFO] Final merged train size: {len(self.train_data)} interactions")
            print(f"[INFO] num_users={self.num_users}, num_items={self.num_items}")
            return

        # ...原本的 source/target domain 處理...
        # ====== Source domain ======
        src_prefix = os.path.join(data_dir, self.source_domain, self.source_domain)
        src_files = [src_prefix + ".train", src_prefix + ".valid", src_prefix + ".test"]

        src_data_list = []
        for f in src_files:
            if os.path.isfile(f):
                df = pd.read_csv(f, sep=sep, header=None, names=columns)
                src_data_list.append(df)
                print(f"[INFO] Loaded {f}, {len(df)} interactions")
            else:
                warnings.warn(f"{f} not found")
        _source_data = pd.concat(src_data_list) if src_data_list else pd.DataFrame()

        # ====== Target domain ======
        tgt_prefix = os.path.join(data_dir, self.target_domain, self.target_domain)
        tgt_train_file = tgt_prefix + ".train"
        tgt_test_file = tgt_prefix + ".test"

        if os.path.isfile(tgt_train_file):
            _target_train = pd.read_csv(tgt_train_file, sep=sep, header=None, names=columns)
            print(f"[INFO] Loaded {tgt_train_file}, {len(_target_train)} interactions")
        else:
            _target_train = pd.DataFrame()
            warnings.warn(f"{tgt_train_file} not found")

        if os.path.isfile(tgt_test_file):
            _target_test = pd.read_csv(tgt_test_file, sep=sep, header=None, names=columns)
            print(f"[INFO] Loaded {tgt_test_file}, {len(_target_test)} interactions")
        else:
            _target_test = pd.DataFrame()
            warnings.warn(f"{tgt_test_file} not found")

        # ====== Item offset (避免 index 衝突) ======
        if not _source_data.empty and not _target_train.empty:
            source_num_items = _source_data[_ITEM].max() + 1
            _target_train[_ITEM] = _target_train[_ITEM] + source_num_items
            _target_test[_ITEM] = _target_test[_ITEM] + source_num_items
            print(f"[INFO] Applied item offset: {self.target_domain} items shifted by {source_num_items}")

        # ====== 合併 ======
        _train_data = pd.concat([_source_data, _target_train])
        _valid_data = pd.DataFrame()
        _test_data = _target_test

        # ====== 統計 ======
        all_data = _train_data
        self.num_users = max(all_data[_USER]) + 1
        self.num_items = max(all_data[_ITEM]) + 1
        self.num_ratings = len(all_data)
        self.num_train_ratings = len(_train_data)

        # ====== 包裝 ======
        self.train_data = Interaction(_train_data, num_users=self.num_users, num_items=self.num_items)
        self.valid_data = Interaction(_valid_data, num_users=self.num_users, num_items=self.num_items)
        self.test_data = Interaction(_test_data, num_users=self.num_users, num_items=self.num_items)

        # ====== 額外 log ======
        self.num_source_items = _source_data[_ITEM].nunique() if not _source_data.empty else 0
        self.num_target_items = _target_train[_ITEM].nunique() if not _target_train.empty else 0
        print(f"[INFO] Final merged train size: {len(self.train_data)} interactions")
        print(f"[INFO] Test size: {len(self.test_data)} interactions")
        print(f"[INFO] num_users={self.num_users}, num_items={self.num_items}")
        print(f"[INFO] num_source_items={self.num_source_items}, num_target_items={self.num_target_items}")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Dataset: {self.data_name}, users={self.num_users}, items={self.num_items}, ratings={self.num_ratings}"
