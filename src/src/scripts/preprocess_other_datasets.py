import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from settings import DIR_TO_ML1M
from collections import defaultdict
from scipy import sparse as sps


class DataLoader:
    def __init__(self, path, n_users_fit=None):

        self.full_matrix, n_users, n_items = self.get_matrix_from_file(os.path.join(path, "train.txt"))
        self.test_matrix, _, _ = self.get_matrix_from_file(os.path.join(path, "test.txt"),
                 n_users=n_users, n_items=n_items
        )

        if isinstance(n_users_fit, int):
            self.full_matrix = self.full_matrix[:n_users_fit]
            self.test_matrix = self.test_matrix[:n_users_fit]
        else:
            n_users_fit = self.full_matrix.shape[0]

        user_split = np.random.choice([0,1,2], size=n_users_fit, p=[0.8, 0.1,0.1],)
        
        self.train_matrix = self.full_matrix[user_split==0,:] + self.test_matrix[user_split==0,:]
        self.valid_matrix_input = self.full_matrix[user_split==1,:]
        self.valid_matrix_true = self.test_matrix[user_split==1,:]
        self.test_matrix_input = self.full_matrix[user_split==2,:]
        self.test_matrix_true = self.test_matrix[user_split==2,:]

        self.n_users, self.n_items = n_users_fit, n_items

    def get_matrix_from_file(self, path, n_users=None, n_items=None):

        if n_users is None and n_items is None:
            n_users = 0
            max_item = 0
            with open(path, "r") as f:
                for line in f:
                    n_users += 1
                    items = list(map(int, line.split()[1:]))
                    max_item = max(max_item, max(items) + 1)

            n_items = max_item

        matrix = np.zeros((n_users, n_items))

        with open(path, "r") as f:
            for line in f:
                line = line.split()
                user = int(line[0])
                items = list(map(int, line[1:]))
                matrix[user, items] = 1

        return matrix, n_users, n_items

    def load_data(self, datatype="train"):
        if datatype == "train":
            return self._load_train_data()
        elif datatype == "validation":
            return self._load_tr_te_data(datatype)
        elif datatype == "test":
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def _load_train_data(self):
        return sps.csr_matrix(self.train_matrix)

    def _load_tr_te_data(self, datatype="test"):

        if datatype == 'valid':
            inp = sps.csr_matrix(self.valid_matrix_input)
            out = sps.csr_matrix(self.valid_matrix_true)
        else:
            inp = sps.csr_matrix(self.valid_matrix_input)
            out = sps.csr_matrix(self.valid_matrix_true)

        return inp, out
