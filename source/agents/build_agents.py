
from smolagents import CodeAgent
from source.tools.synonym_tool import get_synonym


# Create the agents with access to appropriate tools
def create_agents(model,retriever_tool,execute_sql):
    """Create the pipeline agents with access to the library"""
    
    question_generator = CodeAgent(
        model=model,
        tools=[
            #get_tables_info,
            #get_table_samples,
            #check_query_novelty
            retriever_tool
        ],
        name="question_generator",
        description="Generates a new database question based on schema and sample data",
        additional_authorized_imports=["numpy"],max_steps=10
    )

    sql_translator = CodeAgent(
        model=model,
        tools=[execute_sql],
        name="sql_translator", 
        description="Translates natural language questions into SQL queries",
        additional_authorized_imports=["pandas","numpy","time"],max_steps=10#,"sqlite3"
    )

    question_diversity = CodeAgent(
        model=model,
        tools=[get_synonym,retriever_tool],
        name="question_diversity",
        description="Creates diverse variations of questions using different techniques",
        additional_authorized_imports=["pandas","numpy","time"],max_steps=10,
    )
    
    return question_generator, sql_translator, question_diversity