import argparse
import os
import sqlite3


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    return parser


if __name__ == "__main__":

    parser = create_parser()
    args, _ = parser.parse_known_args()
    path = os.path.join(args.path, "results_data.db")

    try:
        sqlite_connection = sqlite3.connect(path)
        sqlite_create_table_query = """CREATE TABLE semi_pers_res (
                                    id INTEGER PRIMARY KEY,
                                    method_name TEXT NOT NULL,
                                    metric_name TEXT NOT NULL,
                                    k_cutoff INTEGER NOT NULL,
                                    dataset_name TEXT NOT NULL,
                                    metric_value REAL NOT NULL,
                                    timestamp unixepoch NOT NULL);"""

        cursor = sqlite_connection.cursor()
        print("База данных подключена к SQLite")
        cursor.execute(sqlite_create_table_query)
        sqlite_connection.commit()
        print("Таблица SQLite создана")

        cursor.close()

    except sqlite3.Error as error:
        print("Ошибка при подключении к sqlite", error)
    finally:
        if sqlite_connection:
            sqlite_connection.close()
            print("Соединение с SQLite закрыто")
