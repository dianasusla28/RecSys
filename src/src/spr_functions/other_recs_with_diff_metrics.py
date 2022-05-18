import numpy as np
from scipy.spatial import distance

from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from spr_functions.base import ParameterizedRecommender
from implicit.als import AlternatingLeastSquares
from hyperopt import fmin, hp, tpe
from scipy import sparse as sps

from sklearn.cluster import KMeans
from spr_functions.get_user_vector import get_als_vectors
from annoy import AnnoyIndex
import bottleneck as bn
import xgboost as xgb


class OtherRecsALSRecommender(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr, loader)

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128)
        }
        return space


    def predict(self, data_tensor):
        
        train_matrix_copy = self.train_matrix.toarray()

        ids_of_similar_users = []

        for user in range(data_tensor.shape[0]):
            if user == 0:
                print(self.metric)
            user_vector = data_tensor[user].detach().cpu().numpy()
            distances = distance.cdist([user_vector], train_matrix_copy, self.metric)[0]
            distances[train_matrix_copy.sum(1) == 0] = np.inf

            min_index = np.argmin(distances)
            ids_of_similar_users.append(min_index)

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.model.user_factors[ids_of_similar_users]
        recs = np.einsum("ud,id->ui", user_vectors, self.model.item_factors)

        return recs

    def fit_with_params(self, hyperparameters) -> float:
        self.metric = hyperparameters['metric']
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train, show_progress=False)

        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric

    
class OtherRecsALSRecommenderKMeans(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix, vad_data_te, vad_data_tr, loader)     

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128)
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
            
            if user == 0:
                print(self.metric)
                    
            distances = distance.cdist([user_emb], train_matrix_copy, self.metric)[0]
            distances[train_matrix_copy.sum(1) == 0] = np.inf

            min_index = np.argmin(distances)
            ids_of_similar_users.append(cluster_ids[min_index])

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.ALSmodel.user_factors[ids_of_similar_users]

        recs = np.einsum("ud,id->ui", user_vectors, self.ALSmodel.item_factors)
        
        return recs

    def fit_with_params(self, hyperparameters) -> float:
        
        self.metric = hyperparameters['metric']
       
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.ALSmodel = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.ALSmodel.fit(copied_train, show_progress=False)
        
        self.KMEANSmodel = KMeans(n_clusters=hyperparameters['n_clusters'])
        self.KMEANSmodel.fit(self.ALSmodel.user_factors.astype('double'))
        
        print('Number of clasters:', len(self.KMEANSmodel.cluster_centers_))

        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric

    
class OtherRecsALSRecommenderKNN(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr, loader)

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128)
        }
        return space


    def predict(self, data_tensor):
        
        train_matrix_copy = self.train_matrix.toarray()

        ids_of_similar_users = []

        for user in range(data_tensor.shape[0]):
            user_vector = data_tensor[user].detach().cpu().numpy()
            
            min_index = self.t.get_nns_by_vector(user_vector, self.K + 1, include_distances=False)[1:]
            ids_of_similar_users.append(min_index)

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.model.user_factors[ids_of_similar_users]

        recs = np.einsum("und,id->ui", user_vectors, self.model.item_factors)
        
        return recs

    def fit_with_params(self, hyperparameters) -> float:
        self.K = hyperparameters['K']
        
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train, show_progress=False)

        f = copied_train.shape[1]
        self.t = AnnoyIndex(f, hyperparameters['metric'])
        for i in tqdm(range(copied_train.shape[0])):
            self.t.add_item(i, copied_train.toarray()[i])

        self.t.build(10)
        
        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric
    
    
class OtherRecsALSRecommenderKNN_topK(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr, loader)

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128)
        }
        return space


    def predict(self, data_tensor):
        
        train_matrix_copy = self.train_matrix.toarray()

        ids_of_similar_users = []

        for user in range(data_tensor.shape[0]):
            user_vector = data_tensor[user].detach().cpu().numpy()
            
            min_index = self.t.get_nns_by_vector(user_vector, self.K + 1, include_distances=False)[1:]
            ids_of_similar_users.append(min_index)

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.model.user_factors[ids_of_similar_users]

        res = np.einsum("und,id->uni", user_vectors, self.model.item_factors)
        
        indices = bn.argpartition(-res, self.topK, axis=-1)

        matrix_with_inf = np.full(res.shape, res.min())
        values = np.take_along_axis(res, indices[:, :, :self.topK], axis=-1)
        np.put_along_axis(matrix_with_inf, indices[:, :, :self.topK], values, axis=-1)
        recs = np.max(matrix_with_inf, axis=1)
        
        
        return recs

    def fit_with_params(self, hyperparameters) -> float:
        self.K = hyperparameters['K']
        self.topK = hyperparameters['topK']
        #self.eval_m = hyperparameters['eval_m']
        
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train, show_progress=False)

        f = copied_train.shape[1]
        self.t = AnnoyIndex(f, hyperparameters['metric'])
        for i in tqdm(range(copied_train.shape[0])):
            self.t.add_item(i, copied_train.toarray()[i])

        self.t.build(10)
        
        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric
    
    
class OtherRecsALSRecommenderKNN_weight(ParameterizedRecommender):

    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr, loader)

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 64, 128)
        }
        return space


    def predict(self, data_tensor):
        
        train_matrix_copy = self.train_matrix.toarray()

        ids_of_similar_users = []
        distances_matrix = []

        for user in range(data_tensor.shape[0]):
            user_vector = data_tensor[user].detach().cpu().numpy()
        
            min_indexes, distances = self.t.get_nns_by_vector(user_vector, self.K + 1, include_distances=True)
        
            min_indexes = min_indexes[1:] 
            distances = distances[1:]
            ids_of_similar_users.append(min_indexes)
            distances_matrix.append(distances)

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.model.user_factors[ids_of_similar_users]

        weight_matrix = 1 / (np.array(distances_matrix) + 1e-8)
        weight_matrix /=  weight_matrix.sum(axis=1, keepdims=True)
    
        interm_recs = np.einsum("und,id->uni", user_vectors, self.model.item_factors)
    
        recs =  np.einsum("uni,un->ui", interm_recs, weight_matrix)
    
        return recs

    def fit_with_params(self, hyperparameters) -> float:
        self.K = hyperparameters['K']
        
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train, show_progress=False)

        f = copied_train.shape[1]
        self.t = AnnoyIndex(f, hyperparameters['metric'])
        for i in tqdm(range(copied_train.shape[0])):
            self.t.add_item(i, copied_train.toarray()[i])

        self.t.build(10)
        
        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric
    
    
    
class OtherRecsALSRecommenderKNN_XGBboost(ParameterizedRecommender):
    def __init__(self, train_data, train_matrix, vad_data_te, vad_data_tr, loader):

        super().__init__(train_data, train_matrix,vad_data_te, vad_data_tr, loader)

    @property
    def model_params(self):

        space = {
            "alpha": hp.randint("alpha", 6, 50),
            "factors": hp.randint("factors", 32, 128),
            'K': hp.randint("K", 25, 26)
        }
        return space


    def predict(self, data_tensor):
        
        train_matrix_copy = self.train_matrix.toarray()

        ids_of_similar_users = []

        for user in range(data_tensor.shape[0]):
            user_vector = data_tensor[user].detach().cpu().numpy()
            
            min_index = self.t.get_nns_by_vector(user_vector, self.K + 1, include_distances=False)[1:]
            ids_of_similar_users.append(min_index)

        ids_of_similar_users = np.array(ids_of_similar_users)
        user_vectors = self.model.user_factors.to_numpy()[ids_of_similar_users]

        res = np.einsum("und,id->uin", user_vectors, self.model.item_factors.to_numpy())
        recs = self.boosting.predict(res.reshape((-1, self.K))).reshape((-1, self.model.item_factors.shape[0]))
        
        return recs

    def fit_with_params(self, hyperparameters) -> float:
        print(hyperparameters)
        self.K = hyperparameters['K']
        
        copied_train = self.train_matrix.copy()
        copied_train.data = 1.0 + hyperparameters['alpha'] * copied_train.data

        self.model = AlternatingLeastSquares(factors=hyperparameters['factors'])

        self.model.fit(copied_train, show_progress=False)

        f = copied_train.shape[1]
        copied_train = copied_train.toarray()
        self.t = AnnoyIndex(f, 'angular')
        for i in tqdm(range(copied_train.shape[0])):
            self.t.add_item(i, copied_train[i])

        self.t.build(10)
        
        dataset = np.empty((copied_train.shape[0], self.model.item_factors.shape[0], self.K), dtype='float32')
        
        for user in range(copied_train.shape[0]):
            #user_vector = copied_train[user].detach().cpu().numpy()
            
            user_vector = copied_train[user]
            
            min_index = self.t.get_nns_by_vector(user_vector, self.K + 1, include_distances=False)[1:]
            user_vectors = self.model.user_factors[min_index]
            # user_vectors: k*d, item_factors: m*d
            # item_factors @ user_vectors.T: m*k
            dataset[user] = self.model.item_factors.to_numpy() @ user_vectors.to_numpy().T

        self.boosting = xgb.XGBRanker(  
            tree_method='gpu_hist',
            booster='gbtree',
            objective='rank:pairwise',
            random_state=42, 
            learning_rate=0.1,
            colsample_bytree=0.7,
            max_depth=6, 
            n_estimators=200, 
            subsample=0.7
        )
        
        self.boosting.fit(dataset.reshape((-1, self.K)), copied_train.reshape(-1), group=np.full(copied_train.shape[0], self.model.item_factors.shape[0]), verbose=True)
        
        valid_metric = self.eval()
        print(valid_metric)
        
        return valid_metric

