import os
import random
import sqlite3
import json
from typing import Dict, Any
from datasets import load_dataset
import pandas as pd


class TableManager:
    """
    TableManager is responsible for retrieving and describing tables from a given database directory.
    Supports CSV, SQLite, and Parquet formats.
    """

    def __init__(self, data_args: str):
        """
        Initialize the TableManager with the database directory.        
        """

        self.prepare_table(data_args)
        self.table_limit = data_args.table_limit
        self.current_table_id = None


    def prepare_db_path(self):
        file = self.squall_table_id_by_id[self.current_table_id] #propre
        print('squall_table_id_by_id',file)
        db_path = f"../data/squall/tables/db/{file}.db"  # Path to the database file
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}. Please ensure the table data exists.")
            
        return db_path

    def get_durty_table(self):

        durty_table = self.wtq_table_by_id[self.current_table_id]
        durty_table = pd.DataFrame(durty_table['rows'], columns=durty_table['header'])

        return durty_table

    def get_clean_table(self):

        clean_table = pd.read_sql_query("SELECT * FROM w", self.conn)

        return clean_table


    def get_random_sampled_tables(self):
        
        clean_df = self.get_clean_table()
        durty_df = self.get_durty_table()


        df_size = clean_df.shape[0]-1
        random_ids = random.sample(range(df_size), min(self.table_limit, df_size))

        table_clean = self.df_to_natural_language_context(clean_df.iloc[random_ids])
        table_durty = self.df_to_natural_language_context(durty_df.iloc[random_ids])

        return table_clean, table_durty


    def df_to_natural_language_context(self, df):

        sentences = []
        for _, row in df.iterrows():
            items = [f"{col}: {row[col]}" for col in df.columns]
            sentence = ", ".join(items) + "."
            sentences.append(sentence)

        context = "Here is a table represented in text:\n\n" + "\n".join(sentences)
        return context


    def prepare_table(self, data_args):
   
        file_path = data_args.squall_path
        with open(file_path, 'r') as json_file:
            squall = json.load(json_file)

        squall_ids = set(i["nt"] for i in squall)
        wtq = load_dataset(data_args.wikitablequestions_path)
        wtq_train = wtq["train"]
        wtq_ids = set(wtq_train["id"])
        common_ids = list(squall_ids.intersection(wtq_ids))
        self.squall_table_id_by_id = {entry["nt"]: entry["tbl"] for entry in squall if entry["nt"] in common_ids}
        self.wtq_table_by_id = {entry["id"]: entry["table"] for entry in wtq_train if entry["id"] in common_ids}
        self.common_ids = common_ids


    def get_tables_info(self, conn) :

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        schemas = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            table_info = cursor.fetchall()
            # Format the schema in a more readable way
            formatted_columns = []
            for col in table_info:
                col_id, name, type_, notnull, default, pk = col
                attributes = []
                if pk:
                    attributes.append("PRIMARY KEY")
                if notnull:
                    attributes.append("NOT NULL")
                if default is not None:
                    attributes.append(f"DEFAULT {default}")
                attr_str = ", ".join(attributes)
                if attr_str:
                    formatted_columns.append(f"{name} ({type_}) - {attr_str}")
                else:
                    formatted_columns.append(f"{name} ({type_})")
            schemas[table] = formatted_columns
        
        return {"tables": tables, "schemas": schemas}


    def connect_to_database(self):

        db_path = self.prepare_db_path()
        print("db_path",db_path)
        conn = sqlite3.connect(db_path)
        self.conn = conn

        return conn
        
