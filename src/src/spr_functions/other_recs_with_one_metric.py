import numpy as np
from scipy.spatial import distance

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from spr_functions.base import ParameterizedRecommender
from implicit.als import AlternatingLeastSquares
from hyperopt import fmin, hp, tpe
from scipy import sparse as sps

from sklearn.cluster import KMeans
from spr_functions.get_user_vector import get_als_vectors


class OtherRecsALSRecommender(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr, loader)

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

        self.model.fit(copied_train, show_progress=False)

        self.sim_c = self.sim[hyperparameters['similarity']]

        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric
    
 
    
class OtherRecsALSRecommenderKMeans(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix, vad_data_te, vad_data_tr, loader)     
        
        self.sim = ['cosine','euclidean']
        
    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128),
            "similarity": hp.choice('similarity', [0, 1])
        }
        return space

    def fill_cluster_ids(self, user_cluster):
        
        cluster_ids = []
        for i in range(len(self.KMEANSmodel.labels_)):
            if self.KMEANSmodel.labels_[i] == user_cluster:
                cluster_ids.append(i)
                
        return np.array(cluster_ids)  

    def predict(self, data_tensor):
        
        ids_of_similar_users = []
        
        for user in range(data_tensor.shape[0]):
            
            user_vector = data_tensor[user].detach().cpu().numpy()
            user_emb = get_als_vectors([user_vector], self.ALSmodel.item_factors)[0]
            user_cluster = self.KMEANSmodel.predict([user_emb.astype('double')])[0]
            cluster_ids = self.fill_cluster_ids(user_cluster)
            train_matrix_copy = self.ALSmodel.user_factors[cluster_ids]
                    
            distances = distance.cdist([user_emb], train_matrix_copy, self.sim_c)[0]
            distances[train_matrix_copy.sum(1) == 0] = np.inf

            min_index = np.argmin(distances)
            ids_of_similar_users.append(cluster_ids[min_index])

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.ALSmodel.user_factors[ids_of_similar_users]

        recs = np.einsum("ud,id->ui", user_vectors, self.ALSmodel.item_factors)
        
        return recs

    def fit_with_params(self, hyperparameters) -> float:

        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.ALSmodel = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.ALSmodel.fit(copied_train, show_progress=False)
        
        self.KMEANSmodel = KMeans()
        self.KMEANSmodel.fit(self.ALSmodel.user_factors.astype('double'))
        
        print('Number of clasters:', len(self.KMEANSmodel.cluster_centers_))

        self.sim_c = self.sim[hyperparameters['similarity']]
        
        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric

 