from collections import defaultdict

from joblib import delayed, Parallel
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import torch
from scipy.spatial import distance
from tqdm import tqdm
from hyperopt import fmin, hp, tpe
from spr_functions.base import ParameterizedRecommender
from implicit.als import AlternatingLeastSquares



class ClusteringPopularRecs(ParameterizedRecommender):

    def __init__(self, train_matrix, train_data, vad_data_te, vad_data_tr):

        super().__init__(train_matrix, train_data,vad_data_te, vad_data_tr)

        self.sim = ['cosine','euclidean']

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128),
            "similarity": hp.choice('similarity', [0, 1])
        }
        return space


    def predict(self, data_tensor):

        train_matrix_copy = self.train_matrix.toarray()

        ids_of_similar_users = []

        for user in range(data_tensor.shape[0]):
            user_vector = data_tensor[user].detach().cpu().numpy()
            distances = distance.cdist([user_vector], train_matrix_copy, self.sim_c)[0]
            distances[train_matrix_copy.sum(1) == 0] = np.inf

            min_index = np.argmin(distances)
            ids_of_similar_users.append(min_index)

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.model.user_factors[ids_of_similar_users]
        recs = np.einsum("ud,id->ui", user_vectors, self.model.item_factors)

        return recs

    def fit_with_params(self, hyperparameters) -> float:

        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train.T, show_progress=False)

        self.sim_c = self.sim[hyperparameters['similarity']]

        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric

