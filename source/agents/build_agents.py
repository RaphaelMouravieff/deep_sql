
from smolagents import CodeAgent
from source.tools.synonym_tool import get_synonym


# Create the agents with access to appropriate tools
def create_agents(model, training_args, retriever_tool, execute_sql):
    """Create the pipeline agents with access to the library"""
    
    question_generator = CodeAgent(
        model=model,
        tools=[
            retriever_tool
        ],
        name="question_generator",
        description="Generates a new database question based on schema and sample data",
        additional_authorized_imports=["numpy"], 
        max_steps=training_args.max_agent_steps
    )

    sql_translator = CodeAgent(
        model=model,
        tools=[execute_sql],
        name="sql_translator", 
        description="Translates natural language questions into SQL queries",
        additional_authorized_imports=["pandas","numpy","time"], 
        max_steps=training_args.max_agent_steps
    )

    question_diversity = CodeAgent(
        model=model,
        tools=[get_synonym,retriever_tool],
        name="question_diversity",
        description="Creates diverse variations of questions using different techniques",
        additional_authorized_imports=["pandas","numpy","time"], 
        max_steps=training_args.max_agent_steps
    )
    
    agents = {
        "question_generator": question_generator,
        "sql_translator": sql_translator,
        "question_diversity": question_diversity
    }
    return agents


