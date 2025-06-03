

from source.tools.sql_executor import SQLExecutorTool
from source.tools.semantic_retriver import SemanticRetrieverTool
from source.tools.synonym_tool import get_synonym

def create_tools(conn, vector_store):
    execute_sql= SQLExecutorTool(conn)  
    retriever_tool = SemanticRetrieverTool(vector_store)
    return {"execute_sql":execute_sql, "retriever_tool":retriever_tool, "get_synonym":get_synonym}