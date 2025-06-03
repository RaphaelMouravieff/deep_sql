
import sqlite3

class SQLExecutor():
    name = "execute_sql"
    description = (
        "Executes an SQLite SQL query to validate and retrieve results. "
        "If the query runs and returns data, it's accepted and returned as-is. "
        "If the query fails or returns nothing, an error is raised. Input must be a string."
    )

    inputs = {
        "sql_query": {
            "type": "string",
            "description": "The SQL query to validate and execute."
        }
    }

    output_type = "string"

    def __init__(self, conn: sqlite3.Connection, **kwargs):
        super().__init__(**kwargs)
        self.conn = conn
        self.empty_result_count = 0      
        self.execution_error_count = 0

    def forward(self, sql_query: str) -> str:
        assert isinstance(sql_query, str), "The SQL query must be a string."

        cursor = self.conn.cursor()

        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()

            if not results:
                self.empty_result_count += 1
                raise ValueError(f"Query returned no results: {sql_query}")

            return results

        except Exception as e:
            self.execution_error_count += 1
            return f"Error executing SQL: {str(e)} | Query: {sql_query}"
        