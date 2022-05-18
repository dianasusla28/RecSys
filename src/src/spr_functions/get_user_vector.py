import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from spr_functions.base import ParameterizedRecommender
from implicit.als import AlternatingLeastSquares
from hyperopt import fmin, hp, tpe
from scipy import sparse as sps


class IALSInductiveRecommender(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr)

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128)
        }
        return space


    def predict(self, data_tensor):

        user_als_vectors = get_als_vectors(user_vectors=data_tensor, item_factors=self.model.item_factors)
        user_vectors = np.array(user_als_vectors)
        recs = np.einsum("ud,id->ui", user_vectors, self.model.item_factors)

        return recs

    def fit_with_params(self, hyperparameters) -> float:
        
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train, show_progress=False)

        valid_metric = self.eval()
        print(valid_metric)
        return valid_metric



def get_als_vectors(user_vectors, item_factors):
    #user_vectors = user_vectors.detach().cpu().numpy()
    
    lambda_val = 0.1
    yTy = item_factors.T.dot(item_factors)  # d * d matrix

    # X = np.random.normal(size=(1, item_factors.shape[1]))
    Y_eye = sparse.eye(item_factors.shape[0])

    lambda_eye = lambda_val * np.ones(
        (item_factors.shape[1], item_factors.shape[1])
    )  # * sparse.eye(item_factors.shape[1])

    # Compute yTy and xTx at beginning of each iteration to save computing time

    result_vectors = []
    for user_vector in user_vectors:
        conf_samp = user_vector.T  # Grab user row from confidence matrix and convert to dense

        pref = conf_samp.copy()
        pref[pref != 0] = 1  # Create binarized preference vector

        CuI = sparse.diags(conf_samp, 0).toarray()  # Get Cu - I term, don't need to subtract 1 since we never added it
        # (d, n_items) * (n_items * n_items) * (n_items, d)
        yTCuIY = item_factors.T.dot(CuI).dot(item_factors)  # This is the yT(Cu-I)Y term

        # (d, n_items) * (n_items, n_items) * (n_items, 1)
        yTCupu = item_factors.T.dot(CuI + Y_eye).dot(
            pref.T
        )  # This is the yTCuPu term, where we add the eye back in
        # Cu - I + I = Cu

        xx = sps.csr_matrix(yTy + yTCuIY + lambda_eye)
        yy = sps.csr_matrix(yTCupu.T)
        X = spsolve(xx, yy)
        result_vectors.append(X)
        # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper
        # Begin iteration to solve for Y based on fixed X

    return result_vectors


def learn_als_vector_on_fly(model, data_tensor, train_matrix):

    if data_tensor.shape[0] == 0:
        return np.ones((0, data_tensor.shape[1]))

    train_matrix = train_matrix.toarray()

    user_als_vectors = get_als_vectors(data_tensor.detach().cpu().numpy(), model.item_factors)

    user_vectors = np.array(user_als_vectors)
    recs = np.einsum("ud,id->ui", user_vectors, model.item_factors)

    return recs
