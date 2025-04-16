from typing import Dict, List, Tuple, Any, Union, Optional
import json
import os
from tqdm import tqdm


from smolagents import CodeAgent, HfApiModel, OpenAIServerModel,LiteLLMModel


from langchain_community.vectorstores import FAISS


from datasets import load_dataset

from source.bin.main_step import run_pipeline_step
from source.tools.tools import get_synonym, ExecuteSQLTool
from source.prompts.prompt import get_extra_prompt_divers, get_prompt, get_extra_prompt_sql
from source.utils.utils_main import get_table_dirty, get_table, init_library, save_library


from source.tools.retriever import RetrieverTool

# Initialize model based on available API keys
if not os.environ.get("OPENAI_API_KEY"):
    #model = HfApiModel()
    model = LiteLLMModel(# qwen2.5:14b
        #deepseek-r1:14b /llama3.1:8b-instruct-fp16 ollama run qwq:32b-fp16
    model_id="ollama_chat/qwen2.5:14b", # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    num_ctx=8192,device="mps" # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
    ,   )   
    
else: 
    model = OpenAIServerModel("gpt-4o")



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




# The main loop for dataset generation
def generate_dataset(db_path: str, table_id:str,num_entries: int, library_path: str = "sql_dataset_library.json",vector_store_path='vector_store') -> None:
    """
    Generate a dataset with the specified number of entries
    
    Args:
        db_path: Path to the SQLite database
        num_entries: Number of entries to generate
        library_path: Path to save the library JSON
    """
    # Initialize or load existing library
    library,vector_store = init_library(library_path,vector_store_path)
    print(f"Starting with library containing {len(library)} entries")
    
    retriever_tool = RetrieverTool(vector_store)


    conn,tables_info,table_samples = get_table(db_path)
    question_prompt,sql_prompt=get_prompt(tables_info,table_samples,library)
    execute_sql= ExecuteSQLTool(conn)  

    # Create agents
    question_generator, sql_translator, question_diversity = create_agents(model,retriever_tool,execute_sql)
    
    # Main generation loop
    progress_bar = tqdm(range(num_entries), desc="Generating dataset entries")
    for i in progress_bar:
        progress_bar.set_description(f"Entry {len(library) + 1}")
        
        # Run one pipeline step
        entries = run_pipeline_step(question_prompt,sql_prompt,tables_info, table_id,
                                    question_generator, sql_translator, question_diversity,
                                    retriever_tool,execute_sql)
        
        if entries:
            # Add to library
            for entry in entries:
                library.append(entry)
            print(f"Added entry #{len(library)} to library")
            
            # Save after each successful addition
            save_library(library,vector_store, library_path,vector_store_path)
            
            progress_bar.set_postfix(library_size=len(library))
        else:
            print("Failed to generate entry, continuing...")
    
    print(f"Dataset generation complete. Final library size: {len(library)}")


def main():
    squall_table_id_by_id, wtq_table_by_id, common_ids= get_table_dirty()
    #results 
    for table_id in common_ids:
        #id0 = common_ids[0] # salle 
        file=squall_table_id_by_id[table_id] #propre
        db_path = f"../data/tables/db/{file}.db"  # Path to the database file
        #db_path ="../data/tables/db/200_0.db"
        num_entries = 10  # Number of entries to generate
        print(file)
        generate_dataset(db_path,table_id, num_entries)


# Example usage
if __name__ == "__main__":
    main()


