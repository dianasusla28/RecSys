import argparse
import os
from re import I
import sqlite3
import time
import bottleneck as bn


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
from spr_functions.other_recs_with_diff_metrics import OtherRecsALSRecommenderKNN, OtherRecsALSRecommenderKNN_weight, OtherRecsALSRecommenderKNN_topK
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


def evaluate(recommender, loader, is_fit=False, n20f=True, n100f=True, r20f=True, r50f=True):
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
        instance = recommender(train_data=train_data,
                           train_matrix=train_matrix,
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
            
            #print('args.batch_size', args.batch_size)
            #print('heldout_data', heldout_data.shape, heldout_data)  
            #print('recon_batch', recon_batch.shape, recon_batch)
            
            if n20f == True:
                n20 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
                n20_list.append(n20)
            if n100f == True:
                n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                n100_list.append(n100)
            if r20f == True:
                r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
                r20_list.append(r20)
            if r50f == True:
                r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)
                r50_list.append(r50)

    res = [] 
    total_loss /= len(range(0, e_N, args.batch_size))
    res.append(total_loss)
    
    if n20f == True:
        res.append(np.concatenate(n20_list).mean())
    if n100f == True:
        res.append(np.concatenate(n100_list).mean())
    if r20f == True:
        res.append(np.concatenate(r20_list).mean())
    if r50f == True:
        res.append(np.concatenate(r50_list).mean())

    return res


if __name__ == "__main__":
    parser = create_parser()
    args, _ = parser.parse_known_args()

    db_path = os.path.join(args.db_path, "results_data.db")
    sqlite_connection = sqlite3.connect(db_path)

    if args.data == "ml-1m":
        loader = DL(args.data)
    else:
        loader = DataLoader(args.data, n_users_fit=args.n_users)
    

    #for K in np.logspace(0.6, 2.2, num=15, dtype=int):
    if args.data == 'ml-1m':
        K = 50
    elif args.data == 'gowalla' or args.data == 'yelp2018':
        K = 25
    elif args.data == 'amazon-book':
        K = 15
    else:
        K = 30
    
    metrics = ['n20', 'n100', 'r20', 'r50']
    for metric in metrics:
        train_matrix = loader._load_train_data()
        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')
        model = OtherRecsALSRecommenderKNN_topK(train_data=train_data,
                                                  train_matrix=train_matrix,
                                                  vad_data_te=vad_data_te,
                                                  vad_data_tr=vad_data_tr,
                                                  loader=loader
                                                 )

        if metric == 'n20':
            topK = 20
        if metric == 'n100':
            topK = 100
        if metric == 'r20':
            topK = 20
        if metric == 'r50':
            topK = 50
            
        hyperparams = {'metric': 'angular', 'alpha': 40, 'factors': 64, 'K': K, 'topK' : topK, 'eval_m' : metric}
    
        model.fit_with_params(hyperparams)
        
        if metric == 'n20':
            res = evaluate(model, loader, True, True, False, False, False)
            test_loss, n20 = res[0], res[1]
            print(n20)
        if metric == 'n100':
            res = evaluate(model, loader, True, False, True, False, False)
            test_loss, n100 = res[0], res[1]
        if metric == 'r20':
            res = evaluate(model, loader, True, False, False, True, False)
            test_loss, r20 = res[0], res[1]
        if metric == 'r50':
            res = evaluate(model, loader, True, False, False, False, True)
            test_loss, r50 = res[0], res[1]

    print("=" * 89)
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
