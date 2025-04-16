from smolagents import Tool
import sqlite3


class SQLExecutorTool(Tool):
    name = "execute_sql"
    description = (
        "Executes an SQL query (SQLite) to check if it's valid and returns results. "
        "SQL query input must be a string. If everything is good the output is the sql query."
    )

    inputs = {
        "sql_query": {
            "type": "string",
            "description": ("The string SQL query code to execute.")
        }
    }
    output_type = "string"  # Could be "object" or "string" depending on your framework

    def __init__(self, conn: sqlite3.Connection, **kwargs):
        """
        Args:
            conn: A live sqlite3.Connection to the target database.
        """
        super().__init__(**kwargs)  # Optional, for compatibility with base class
        self.conn = conn

    def execute_it(self,sql_query:str)-> str:
        """
        Executes the provided SQL query and returns the results. 
        test before we add it to lib
        """

        assert isinstance(sql_query, str), "Your SQL query must be a string."
        cursor = self.conn.cursor()
        cursor.execute(sql_query)
        return (cursor.fetchall())



    def forward(
            self, 
            sql_query: str
            ) -> str :
        """
        Executes the provided SQL query and returns the results.

        Args:
            sql_query (str): The SQL query to run.

        Returns:
            List[Tuple]: The raw results of the query. If an error occurs,
                         returns a single-element list with an error message tuple.
        """
        assert isinstance(sql_query, str), "Your SQL query must be a string."
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(sql_query)
            answ= cursor.fetchall()
            
            
            if len(answ)==0:raise ValueError(f"Wrong, no results it returns an empty list")
            print(f"Good job. The sql query is accepted and the results are "+str(answ))
            return sql_query
        except Exception as e:
            return f"Error executing SQL: {str(e)} for the query: {sql_query}"