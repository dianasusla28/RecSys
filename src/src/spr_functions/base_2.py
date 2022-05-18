import torch

from hyperopt import fmin, hp, tpe
import lightgbm
from loguru import logger
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from scripts.main import naive_sparse2tensor
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
from tqdm import tqdm



class BaseInductiveRecommender:

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        self.train_data=train_data
        self.train_matrix=train_matrix
        self.vad_data_te=vad_data_te
        self.vad_data_tr=vad_data_tr
        self.loader=loader

    def prepare(self):
        raise NotImplementedError


    def predict(self, data_tensor):
        raise NotImplementedError


class ParameterizedRecommender(BaseInductiveRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):
        super().__init__(train_data, train_matrix, vad_data_te, vad_data_tr, loader)

        self.batch_size = 128 
        self.num_cases = 10


    def prepare(self):
        best_params = self.get_best_params()
        print(best_params)
        self.fit_with_params(best_params)


    def fit_with_params(self, hyperparameters: dict) -> float:
        pass
    
    def get_best_params(self) -> dict:

        space = self.model_params
        print(space)
        # minimize the objective over the space
        best = fmin(
            lambda x: -self.fit_with_params(hyperparameters=x),
            space,
            algo=tpe.suggest,
            max_evals=self.num_cases,
        )

        return best

    def eval(self):

        e_idxlist = list(range(self.vad_data_tr.shape[0]))
        e_N = self.vad_data_tr.shape[0]

        n20_list = []
        n100_list = []
        r20_list = []
        r50_list = []


        count = 0
        with torch.no_grad():
            for start_idx in tqdm(range(0, e_N, self.batch_size)):
                data = self.vad_data_tr[e_idxlist[start_idx : start_idx + self.batch_size]]

                heldout_data = self.vad_data_te[e_idxlist[start_idx : start_idx + self.batch_size]]

                data_tensor = naive_sparse2tensor(data)
                count += data_tensor.shape[0]

                # Exclude examples from training set

                recon_batch = self.predict(data_tensor=data_tensor)
                recon_batch[data.nonzero()] = -np.inf
        

                if self.eval_m == 'n20':
                    n20 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
                    n20_list.append(n20)
                if self.eval_m == 'n100':
                    n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                    n100_list.append(n100)
                if self.eval_m == 'r20':
                    r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
                    r20_list.append(r20)
                if self.eval_m == 'r50':
                    r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)
                    r50_list.append(r50)

                    
        if self.eval_m == 'n20':
            res = np.concatenate(n20_list)
        if self.eval_m == 'n100':
            res = np.concatenate(n100_list)
        if self.eval_m == 'r20':
            res = np.concatenate(r20_list)
        if self.eval_m == 'r50':
            res = np.concatenate(r50_list)

        return np.mean(res)


def evaluate(recommender, train_matrix, data_tr, data_te, vad_data_tr, vad_data_te):
    # Turn on evaluation mode
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    n20_list = []
    n100_list = []
    r20_list = []
    r50_list = []

    instance = recommender(train_matrix=train_matrix,
                            vad_data_te=vad_data_te,
                            vad_data_tr=vad_data_tr)
    instance.prepare()

    count = 0
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):

            data = data_tr[e_idxlist[start_idx : start_idx + args.batch_size]]
            heldout_data = data_te[e_idxlist[start_idx : start_idx + args.batch_size]]

            data_tensor = naive_sparse2tensor(data)
            count += data_tensor.shape[0]

            # Exclude examples from training set

            recon_batch = instance.predict(data_tensor=data_tensor)
            recon_batch[data.nonzero()] = -np.inf

            n20 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n20_list.append(n20)
            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    total_loss /= len(range(0, e_N, args.batch_size))
    n20_list = np.concatenate(n20_list)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n20_list), np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)
