import numpy as np
import torch
from spr_functions.base import BaseInductiveRecommender, ParameterizedRecommender
from hyperopt import hp
from utils.similarity import Compute_Similarity_Python
from scipy import sparse as sps



class RandomInductiveRecommender(BaseInductiveRecommender):
    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr):
        super().__init__(train_data=train_data, train_matrix=train_matrix, vad_data_te=vad_data_te, vad_data_tr=vad_data_tr)

    def prepare(self):
        pass

    def predict(self, data_tensor):
        return torch.rand(size=data_tensor.shape)


class TopPopularInductiveRecommender(BaseInductiveRecommender):
    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr):
        super().__init__(train_data=train_data, train_matrix=train_matrix, vad_data_te=vad_data_te, vad_data_tr=vad_data_tr)

    def prepare(self):
        self.scores = self.train_matrix.toarray().sum(0)
        pass

    def predict(self, data_tensor):
        return np.tile(self.scores, (data_tensor.shape[0], 1))


class UserKNNRecommender(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr):
        super().__init__(train_matrix=train_matrix,train_data=train_data, vad_data_te=vad_data_te, vad_data_tr=vad_data_tr)

    def prepare(self):
        self.scores = self.train_matrix.toarray().sum(0)
        pass

    def predict(self, data_tensor):
        pred = np.tile(self.scores, (data_tensor.shape[0], 1))
        print('predict shape', pred.shape)
        print('predict:', pred)
        return pred


        self.sim = ['cosine','euclidean']


    @property
    def model_params(self):

        space = {
            "similarity": hp.choice('similarity', [0, 1])
        }
        return space


    def predict(self, data_tensor):

        train_matrix_copy = self.train_matrix.toarray()
        extended_matrix = sps.csr_matrix(np.concatenate((train_matrix_copy, data_tensor)))
        user_weights = Compute_Similarity_Python(extended_matrix.T).compute_similarity()
        user_weights = user_weights[:data_tensor.shape[0],data_tensor.shape[0]:]
        predictions = user_weights.dot(train_matrix_copy)
        assert predictions.shape == data_tensor.shape
        
        return predictions


    def fit_with_params(self, hyperparameters) -> float:

        self.sim_c = self.sim[hyperparameters['similarity']]

        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric

