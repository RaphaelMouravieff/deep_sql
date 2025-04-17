
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
def generate_dataset(model, data_args, training_args, db_path: str, table_id:str, library_path: str = "sql_dataset_library.json",vector_store_path='vector_store') -> None:

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
    progress_bar = tqdm(range(training_args.num_iterations), desc="Generating dataset entries")
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

    squall_table_id_by_id, wtq_table_by_id, common_ids= get_table_dirty(data_args)
    
    model = load_model(model_args)

    for table_id in common_ids:
        file=squall_table_id_by_id[table_id] #propre
        db_path = f"../data/tables/db/{file}.db"  # Path to the database file
        print(file)
        generate_dataset(model, data_args, training_args, db_path, table_id)


# Example usage
if __name__ == "__main__":
    main()


