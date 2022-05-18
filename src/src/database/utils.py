

def insert_value(sql_instance, result_dict):

    rd = result_dict

    cursor = sql_instance.cursor()
    assert set(result_dict.keys()) <= set(
        ["method_name", "metric_name", "k_cutoff", "dataset_name", "metric_value", "timestamp"]
    )

    sqlite_insert_query = """INSERT INTO semi_pers_res
     (id, method_name, metric_name, k_cutoff, dataset_name, metric_value, timestamp) 
    VALUES  (NULL, ?,?,?,?,?,?)""".format(
        **result_dict
    )

    values = [
        rd["method_name"],
        rd["metric_name"],
        rd["k_cutoff"],
        rd["dataset_name"],
        rd["metric_value"],
        rd["timestamp"],
    ]

    count = cursor.execute(sqlite_insert_query, values)
    sql_instance.commit()
    cursor.close()