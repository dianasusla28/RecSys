import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from spr_functions.base import ParameterizedRecommender
from implicit.als import AlternatingLeastSquares
from hyperopt import fmin, hp, tpe
from joblib import delayed, Parallel
from lightgbm import LGBMClassifier
from sklearn.preprocessing import normalize
from scipy import sparse as sps


class EASEInductiveRecommender(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix, vad_data_te, vad_data_tr, loader)

        self.num_cases=10

    @property
    def model_params(self):

        space = {
            "l2_norm": hp.uniform('l2_norm', 0, 10000),
        }
        return space


    def predict(self, data_tensor):
        return data_tensor @ self.item_similarity

    def fit_with_params(self, hyperparameters) -> float:

        reg_weight = hyperparameters['l2_norm']

        X = self.train_matrix.copy()

        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)

        # gram matrix
        G = X.T @ X

        # add reg to diagonal
        G += reg_weight * sps.identity(G.shape[0])

        # convert to dense because inverse will be dense
        G = G.todense()

        # invert. this takes most of the time
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        # zero out diag
        np.fill_diagonal(B, 0.)

        # instead of computing and storing the entire score matrix,
        # just store B and compute the scores on demand
        # more memory efficient for a larger number of users
        # but if there's a large number of items not much one can do:
        # still have to compute B all at once
        # S = X @ B
        # self.score_matrix = torch.from_numpy(S).to(self.device)

        # torch doesn't support sparse tensor slicing,
        # so will do everything with np/scipy
        self.item_similarity = B
        self.interaction_matrix = X    

        valid_metric = self.eval()
        return valid_metric
