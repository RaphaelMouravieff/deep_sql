

from source.tools.sql_executor import SQLExecutorTool
from source.tools.semantic_retriver import SemanticRetrieverTool
from source.tools.synonym_tool import get_synonym
from source.tools.answer_check import AnswerCheckTool

def create_tools(conn, vector_store, model_args, logger):
    execute_sql= SQLExecutorTool(conn)  
    retriever_tool = SemanticRetrieverTool(vector_store)

    if model_args.use_model_check:
        logger.info('Loading model for answer checking...')
        from source.models.ft_model_setup import load_model, load_tokenizer, load_config
        config = load_config(model_args, logger)
        tokenizer = load_tokenizer(model_args, logger)
        model = load_model(model_args, config, logger)

        answer_check = AnswerCheckTool(model, tokenizer, config)
    return {"execute_sql":execute_sql, "retriever_tool":retriever_tool, "get_synonym":get_synonym}