import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from spr_functions.base import ParameterizedRecommender
from implicit.als import AlternatingLeastSquares
from hyperopt import fmin, hp, tpe
from joblib import delayed, Parallel
from lightgbm import LGBMClassifier



class XGBoostInductiveRecommender(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr):

        super().__init__(train_data, train_matrix, vad_data_te, vad_data_tr)

        self.train_matrix = self.train_matrix.toarray()

    @property
    def model_params(self):

        space = {
            "num_leaves": hp.randint("num_leaves", 6, 50),
        }
        return space


    
    def get_train_table(self, idx, matrix):

        target_table = matrix[:, idx]
       # print(matrix)
        if idx != matrix.shape[1] - 1:
            feature_table = np.concatenate((matrix[:, :idx], matrix[:, idx + 1 :]), axis=1)
        else:
            feature_table = matrix[:, :idx]

        #print(feature_table)
        return feature_table, target_table

    def get_model(self, idx,**hyperparameters):

        features, target_table = self.get_train_table(idx=idx, matrix=self.train_matrix)
        model = LGBMClassifier(**hyperparameters, n_jobs=1, n_estimators=3)
        model.fit(features, target_table)
        return model


    def predict(self, data_tensor):

        scores = []
        for i in tqdm(range(data_tensor.shape[1])):
            features, _ = self.get_train_table(idx=i, matrix=data_tensor)
            scores.append(self.models[i].predict_proba(features)[:, 1])
        scores = np.array(scores).T
        return scores


    def fit_with_params(self, hyperparameters) -> float:


        self.n_items = self.train_matrix.shape[1]
        self.models = []
        print("fitting")
        self.models = Parallel(n_jobs=2)(
            delayed(self.get_model)(i,**hyperparameters) for i in tqdm(range(self.train_matrix.shape[1]))
        )
        print("done")

        valid_metric = self.eval()
        print(valid_metric)
        return valid_metric
