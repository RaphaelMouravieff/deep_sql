
from tqdm import tqdm

from transformers import HfArgumentParser


from source.bin.main_step import run_pipeline_step
from source.prompts.prompt import get_prompt
from source.library.tables import get_table_dirty, get_table
from source.library.storage import  init_library, save_library

from source.agents.build_agents import create_agents

from source.tools.sql_executor import SQLExecutorTool

from source.tools.semantic_retriver import SemanticRetrieverTool


from source.models.model_setup import load_model

from source.utils.args import  ModelArguments, DataArguments, TrainingArguments


# The main loop for dataset generation
def generate_dataset(db_path: str, table_id:str,num_entries: int, model, library_path: str = "sql_dataset_library.json",vector_store_path='vector_store') -> None:
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
    
    retriever_tool = SemanticRetrieverTool(vector_store)


    conn,tables_info,table_samples = get_table(db_path)
    question_prompt,sql_prompt=get_prompt(tables_info,table_samples,library)

    execute_sql= SQLExecutorTool(conn)  

    # Create agents
    question_generator, sql_translator, question_diversity = create_agents(model, retriever_tool,execute_sql)
    
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

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    squall_table_id_by_id, wtq_table_by_id, common_ids= get_table_dirty()
    
    model = load_model(model_args)

    for table_id in common_ids:
        #id0 = common_ids[0] # salle 
        file=squall_table_id_by_id[table_id] #propre
        db_path = f"../data/tables/db/{file}.db"  # Path to the database file
        #db_path ="../data/tables/db/200_0.db"
        num_entries = 10  # Number of entries to generate
        print(file)
        generate_dataset(db_path,table_id, num_entries, model)


# Example usage
if __name__ == "__main__":
    main()


