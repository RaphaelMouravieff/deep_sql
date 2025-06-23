
from smolagents import CodeAgent



# Create the agents with access to appropriate tools
def create_agents(model, data_args, tools_dic):
    """Create the pipeline agents with access to the library"""
    
    question_generator = CodeAgent(
        model=model,
        tools=[
            tools_dic["retriever_tool"]
        ],
        name="question_generator",
        description="Generates a new database question based on schema and sample data",
        additional_authorized_imports=["numpy"], 
        max_steps=data_args.max_agent_steps
    )

    sql_translator = CodeAgent(
        model=model,
        tools=[tools_dic["execute_sql"]],
        name="sql_translator", 
        description="Translates natural language questions into SQL queries",
        additional_authorized_imports=["pandas","numpy","time"], 
        max_steps=data_args.max_agent_steps
    )

    question_diversity = CodeAgent(
        model=model,
        tools=[tools_dic["get_synonym"]],
        name="question_diversity",
        description="Creates diverse variations of questions using different techniques",
        additional_authorized_imports=["pandas","numpy","time", "random", "re"], 
        max_steps=data_args.max_agent_steps
    )
    
    agents = {
        "question_generator": question_generator,
        "sql_translator": sql_translator,
        "question_diversity": question_diversity
    }
    return agents


