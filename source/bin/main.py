
from tqdm import tqdm
from transformers import HfArgumentParser
import json
import os

from source.prompts.prompt import  PromptManager
from source.library.tables import TableManager
from source.utils.args import  (ModelArguments,
                               DataArguments,
                               TrainingArguments)

from source.bin.main_step import run_pipeline_step
from source.library.storage import  init_library, save_library
from source.agents.build_agents import create_agents
from source.tools.build_tools import create_tools
from source.models.model_setup import load_model

import gc
import time


def generate_dataset(model, data_args, training_args, table_manager, library, vector_store, metrics) -> None:

    

    conn = table_manager.connect_to_database()
    prompt_manager = PromptManager(data_args, table_manager, library)

    tools = create_tools(conn, vector_store)
    agents = create_agents(model, training_args, tools)



    for i in range(training_args.num_iterations):
        entries = run_pipeline_step(prompt_manager, 
                                    agents,
                                    tools)
        
        if entries:
            for entry in entries:
                library.append(entry)

            metrics["retriever_too_similar_count"] += tools["retriever_tool"].too_similar_count
            metrics["sql_empty_result_count"] += tools["execute_sql"].empty_result_count
            metrics["sql_execution_error_count"] += tools["execute_sql"].execution_error_count

            print(
                f"Added entry #{len(library)} to library\n"
                f" - Retriever too_similar_count: {tools['retriever_tool'].too_similar_count}\n"
                f" - SQL empty_result_count: {tools['execute_sql'].empty_result_count}\n"
                f" - SQL execution_error_count: {tools['execute_sql'].execution_error_count}"
            )


            print(
                f"Added entry #{len(library)} to library\n"
                f" - Retriever too_similar_count: {tools['retriever_tool'].too_similar_count}\n"
                f" - SQL empty_result_count: {tools['execute_sql'].empty_result_count}\n"
                f" - SQL execution_error_count: {tools['execute_sql'].execution_error_count}"
            )

            save_library(data_args, library, vector_store)
            break

        else:
            print("Failed to generate entry, continuing...")
    
    print(f"Dataset generation complete. Final library size: {len(library)}")
    del tools
    del agents
    del prompt_manager
    del conn
    gc.collect()

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    table_manager = TableManager(data_args)
    model = load_model(model_args)

    metrics = {
        "retriever_too_similar_count": 0,
        "sql_empty_result_count": 0,
        "sql_execution_error_count": 0
    }
    
    library, vector_store = init_library(data_args, training_args)
    print(f"Starting with library containing {len(library)} entries")

    start_time = time.time()

    total = len(table_manager.common_ids)
    for idx, table_id in enumerate(table_manager.common_ids):
        
        new_start_time = time.time()

        table_manager.current_table_id = table_id
        generate_dataset(model, data_args, training_args, table_manager, library, vector_store, metrics)

        end_time = time.time()
        duration = end_time - new_start_time
        total_duration = end_time - start_time
        print(f"step {idx}/{total} done in {duration:.2f} seconds, total {total_duration:.2f} seconds.\n")



    os.makedirs("../data", exist_ok=True)
    with open("../data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()


