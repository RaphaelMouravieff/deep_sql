import os
import random
import sqlite3
import json
from typing import Dict, Any
from datasets import load_dataset


def connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database and return connection object.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    return conn
    

def get_table_dirty(data_args):
   
    file_path = data_args.squall_path
    with open(file_path, 'r') as json_file:
        squall = json.load(json_file)

    squall_ids = set(i["nt"] for i in squall)
    wtq = load_dataset(data_args.wikitablequestions_path)
    wtq_train = wtq["train"]
    wtq_ids = set(wtq_train["id"])
    common_ids = list(squall_ids.intersection(wtq_ids))
    squall_table_id_by_id = {entry["nt"]: entry["tbl"] for entry in squall if entry["nt"] in common_ids}
    wtq_table_by_id = {entry["id"]: entry["table"] for entry in wtq_train if entry["id"] in common_ids}
    return squall_table_id_by_id, wtq_table_by_id, common_ids


def get_table(db_path):
    conn = connect_to_database(db_path)
    tables_info = get_tables_info(conn)
    table_samples = get_random_table_samples(conn)

    return conn,tables_info,table_samples


def get_tables_info(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get all table names and their schemas from the database in a readable format.
    
    Args:
        conn: SQLite database connection
    
    Returns:
        Dictionary containing tables and their formatted schemas
    """
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


def get_random_table_samples(conn: sqlite3.Connection, limit: int = 10) -> Dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    samples = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {limit};")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            samples[table] = [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            samples[table] = {"error": str(e)}
    
    return samples


def get_table_samples(conn: sqlite3.Connection, limit: int = 5) -> Dict[str, Any]:
    """
    Get sample rows from each table in the database in a readable format.
    
    Args:
        conn: SQLite database connection
        limit: Maximum number of rows to fetch per table
    
    Returns:
        Dictionary with formatted table samples
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    samples = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT {limit};")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Format the sample data as readable rows
            formatted_rows = []
            for row in rows:
                formatted_row = {}
                for i, value in enumerate(row):
                    formatted_row[columns[i]] = value
                formatted_rows.append(formatted_row)
                
            samples[table] = formatted_rows
        except Exception as e:
            samples[table] = {"error": str(e)}
    
    return samples


class TableManager:
    """
    TableManager is responsible for retrieving and describing tables from a given database directory.
    Supports CSV, SQLite, and Parquet formats.
    """

    def __init__(self, db_directory: str):
        """
        Initialize the TableManager with the database directory.
        
        :param db_directory: Path to the directory containing database tables.
        """
        self.db_directory = db_directory
        self.table_paths = self._get_table_paths()

    def _get_table_paths(self) -> list:
        """
        Retrieve all file paths from the database directory.

        :return: List of full file paths of tables.
        """
        if not os.path.exists(self.db_directory):
            raise FileNotFoundError(f"Database directory '{self.db_directory}' does not exist.")

        return [
            os.path.join(self.db_directory, f) 
            for f in os.listdir(self.db_directory) 
            if os.path.isfile(os.path.join(self.db_directory, f))
        ]

    def get_random_table(self) -> str:
        """
        Select a random table file from the database directory.

        :return: The file path of the selected table.
        """
        
        return random.choice(self.table_paths)

    def get_table_info(self, database_file: str) -> str:
        """
        Return a formatted description of a table for prompting.

        :param database_file: Path to the SQLite database file.
        :return: String containing the table name and schema details.
        """
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()

        # Get table names
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        if not tables:
            conn.close()
            raise ValueError(f"No tables found in the database: {database_file}")

        table_name = tables[0][0]  # Load the first table (modify if needed)

        # Get column metadata
        columns_info = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
        conn.close()

        # Format description
        table_description = f"📋 **Table Name:** {table_name}\n"
        table_description += "(column_index, column_name, data_type, not_null, default_value, primary_key)\n"
        for col in columns_info:
            table_description += f"{col}\n"

        return table_description

    def get_random_table_info(self) -> str:
        """
        Select a random table and return its description.

        :return: Formatted table schema description.
        """


        table_path = self.get_random_table()
        return self.get_table_info(table_path), table_path