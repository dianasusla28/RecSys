import argparse
import os
from re import I
import sqlite3
import time


from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
import numpy as np
from recommenders.itemknn import EASEInductiveRecommender
from scripts.main import naive_sparse2tensor

from recommenders.multivae import VAEInductiveRecommender
from scripts.preprocess_other_datasets import DataLoader
from recommenders.xgboostknn import XGBoostInductiveRecommender
from spr_functions.get_user_vector import IALSInductiveRecommender
from spr_functions.other_recs_with_one_metric import OtherRecsALSRecommender, OtherRecsALSRecommenderKMeans
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


def evaluate(recommender, loader):
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

    instance = recommender(train_data=train_data,
                           train_matrix=train_matrix,
                           vad_data_te=vad_data_te,
                           vad_data_tr=vad_data_tr,
                           loader=loader)
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


if __name__ == "__main__":
    parser = create_parser()
    args, _ = parser.parse_known_args()

    db_path = os.path.join(args.db_path, "results_data.db")
    sqlite_connection = sqlite3.connect(db_path)
    
    loader = DataLoader(args.data, n_users_fit=args.n_users)

    # Run on test data.

    implemented_methods = [
        #OtherRecsALSRecommender,
        #RandomInductiveRecommender,
        #TopPopularInductiveRecommender,
        #UserKNNRecommender,
        #OtherRecsALSRecommenderKMeans
        VAEInductiveRecommender,
        #EASEInductiveRecommender,
        #ClusteringPopularRecs,
        #IALSInductiveRecommender,
    ]

    for method in implemented_methods:
        test_loss, n20, n100, r20, r50 = evaluate(
           method, loader 
        )
        print("=" * 89)
        print(
            "| Func: {}| n20 {:4.6f} | n100 {:4.6f} | r20 {:4.6f} | "
            "r50 {:4.6f}".format(method.__name__, n20, n100, r20, r50)
        )
        print("=" * 89)

        for metric, value in zip(
            ["ndcg@20", "ndcg@100", "recall@20", "recall@50"], (n20, n100, r20, r50)
        ):
            result_dict = {
                "method_name": method.__name__,
                "metric_name": metric.split("@")[0],
                "k_cutoff": metric.split("@")[1],
                "dataset_name": "ml_1m",
                "metric_value": value,
                "timestamp": int(time.time()),
            }

            insert_value(sqlite_connection, result_dict)
            
 