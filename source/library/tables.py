import os
import random
import sqlite3
import json
from typing import Dict, Any
from datasets import load_dataset



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
        db_path = f"../data/tables/db/{file}.db"  # Path to the database file
        return db_path


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


    def get_random_table_samples(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        samples = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {self.table_limit};")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                samples[table] = [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                samples[table] = {"error": str(e)}
        
        return samples


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


    def get_table(self):

        
        tables_info = self.get_tables_info(self.conn)
        table_samples = self.get_random_table_samples(self.conn)

        return tables_info, table_samples


    def connect_to_database(self):

        db_path = self.prepare_db_path()

        conn = sqlite3.connect(db_path)
        self.conn = conn
        return conn
    



