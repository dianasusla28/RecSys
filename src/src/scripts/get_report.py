import argparse
import os
import sqlite3

init_readme = """

# oos-rs
Semi-personalized recsys
```
cd spr/src
PYTHONPATH=. poetry run python scripts/preprocess_ml1m.py --data=PATH_TO_ML1M
PYTHONPATH=. poetry run python  scripts/test_your_function.py --data=PATH_TO_ML1M --db_path=../results/
PYTHONPATH=. poetry run python  scripts/get_report.py --db_path=../results/
```
\n\n\n\n\n

## Results: 

"""


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    return parser


if __name__ == "__main__":

    parser = create_parser()
    args, _ = parser.parse_known_args()
    path = os.path.join(args.db_path, "results_data.db")

    sqlite_connection = sqlite3.connect(path)

    cur = sqlite_connection.cursor()
    cur.execute(
        """
    with temp_table as (select method_name, metric_name, k_cutoff, dataset_name, max(id) as max_id
                    from semi_pers_res
                    group by 1,2,3,4
    )
    select *
    from semi_pers_res
    where id in (select max_id from temp_table)
    
    """
    )

    rows = cur.fetchall()
    cur.close()

    rd = {}
    algs_rec50 = []
    for row in rows:
        rd[(row[1], f"{row[2]}_{row[3]}", row[4])] = row[5]
        if f"{row[2]}_{row[3]}" == "recall_50":
            algs_rec50.append((row[1], row[5]))

    metrics = list(set([f"{x[2]}_{x[3]}" for x in rows]))
    datasets = set([x[4] for x in rows])
    algs = [x[0] for x in sorted(algs_rec50, key=lambda x: x[1])]

    
    metrics = sorted(metrics, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
    print(metrics)

    readme = init_readme
    for dataset in datasets:
        readme += f"## {dataset}\n"
        readme += "| | " + "| ".join(metrics) + " |\n"
        readme += "|" + "| ".join(["---"] * (len(metrics) + 1)) + " |\n| "

        for alg in algs:
            readme += f"{alg} | "
            for metric in sorted(metrics):
                v = round(rd[(alg, metric, dataset)], 2)
                readme += f"  {v} | "
            readme += "\n"

        readme += "\n\n\n"

    with open("../README.md", "w") as f:
        print(readme, file=f)
