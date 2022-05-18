import argparse
import os
from re import I
import sqlite3
import time


from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
import numpy as np
from recommenders.itemknn import EASEInductiveRecommender
from scripts.main import naive_sparse2tensor

from recommenders.multivae import MultiVAE, loss_function
from scripts.preprocess_other_datasets import DataLoader
from scripts.preprocess_dataset import DataLoader as DL
from recommenders.xgboostknn import XGBoostInductiveRecommender
from spr_functions.get_user_vector import IALSInductiveRecommender
from spr_functions.other_recs_with_diff_metrics import OtherRecsALSRecommender, OtherRecsALSRecommenderKMeans
from spr_functions.other_recs_with_diff_metrics import OtherRecsALSRecommenderKNN, OtherRecsALSRecommenderKNN_weight, OtherRecsALSRecommenderKNN_topK, OtherRecsALSRecommenderKNN_XGBboost
from spr_functions.simple import (
    RandomInductiveRecommender,
    TopPopularInductiveRecommender,
    UserKNNRecommender
)
from spr_functions.clustering import ClusteringPopularRecs

from database.utils import insert_value
import torch


def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=False, default='../model.pt')
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument(
        "--data", type=str, default="ml-1m", help="Movielens-1m dataset location"
    )
    parser.add_argument("--batch_size", type=int, default=500, help="batch size")
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--n_users", type=int, required=False)
    
    return parser


def evaluate(recommender, loader, is_fit=False):
    train_matrix = loader._load_train_data()
    
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    data_tr, data_te = loader.load_data('test')
    
    # Turn on evaluation mode
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    n20_list = []
    n100_list = []
    r20_list = []
    r50_list = []

    if is_fit == True:
        instance = recommender
    else:
        instance = OtherRecsALSRecommenderKNN_XGBboost(train_data=train_data,
                                                       train_matrix=train_matrix,
                                                       vad_data_te=vad_data_te,
                                                       vad_data_tr=vad_data_tr,
                                                       loader=loader
                                                       )
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
            
            #print('args.batch_size', args.batch_size)
            #print('heldout_data', heldout_data.shape, heldout_data)  
            #print('recon_batch', recon_batch.shape, recon_batch)
            
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


if __name__ == "__main__":
    parser = create_parser()
    args, _ = parser.parse_known_args()

    db_path = os.path.join(args.db_path, "results_data.db")
    sqlite_connection = sqlite3.connect(db_path)

    if args.data == "ml-1m":
        loader = DL(args.data)
    else:
        loader = DataLoader(args.data, n_users_fit=args.n_users)
    

    n20s = []
    n100s = []
    r20s = []
    r50s = []
    #for K in np.logspace(0.6, 2.2, num=15, dtype=int):
    if args.data == 'ml-1m':
        Ks = [55]
    elif args.data == 'gowalla':
        Ks = [20]
    elif args.data == 'yelp2018':
        Ks = [25]
    elif args.data == 'amazon-book':
        Ks = [15]
    else:
        Ks = [30]
    
    # Ks = np.logspace(0.6, 2.2, 15)
    for K in Ks:
        train_matrix = loader._load_train_data()
        
        # check sparsity of matrix
        print(train_matrix.shape)
        nonzero_el = train_matrix.count_nonzero()
        all_el = train_matrix.shape[0]*train_matrix.shape[1]
        print('Sparsity:', (all_el - nonzero_el) / all_el)
    
        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')
        model = OtherRecsALSRecommenderKNN_XGBboost(train_data=train_data,
                                                  train_matrix=train_matrix,
                                                  vad_data_te=vad_data_te,
                                                  vad_data_tr=vad_data_tr,
                                                  loader=loader
                                                 )
        hyperparams = {'metric': 'angular', 'alpha': 40, 'factors': 64, 'K': K, 'topK': 100}
        print(hyperparams)
        model.fit_with_params(hyperparams)
        print(hyperparams)
        test_loss, n20, n100, r20, r50 = evaluate(model, loader, True)
        print(hyperparams)
  
        n20s.append(n20)
        n100s.append(n100)
        r20s.append(r20)
        r50s.append(r50)
    
        print("=" * 89)
        print('K_neighbors:', hyperparams['K'])
        print('Similarity function:', hyperparams['metric'])
        print(
            "| Func: {}| n20 {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | "
            "r50 {:4.4f}".format('OtherRecsALSRecommenderKNN', n20, n100, r20, r50)
        )
        print("=" * 89)

        for metric, value in zip(
            ["ndcg@20", "ndcg@100", "recall@20", "recall@50"], (n20, n100, r20, r50)
        ):
            result_dict = {
                "method_name": 'OtherRecsALSRecommenderKNN',
                "metric_name": metric.split("@")[0],
                "k_cutoff": metric.split("@")[1],
                "dataset_name": "ml_1m",
                "metric_value": value,
                "timestamp": int(time.time()),
            }

            insert_value(sqlite_connection, result_dict)
            
    print('n20:', n20s)
    print('n100:', n100s)
    print('r20:', r20s)
    print('r50:', r50s)