from tqdm import tqdm
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, TapexTokenizer, BartForConditionalGeneration
import json
import os
import time

from source.step1.verification import preprocess_for_verification, evaluate_example
import torch
from source.step1.init_model import load_models

from source.prompts.prompt import  PromptManager
from source.library.tables import TableManager
from source.utils.args import  (ModelArguments,
                               DataArguments)

from source.bin.main_step_likehood import run_pipeline_step
from source.library.storage import  init_library, save_library
from source.agents.build_agents import create_agents
from source.tools.build_tools import create_tools
from source.models.model_setup import load_model

import gc
import time

def generate_dataset(model, model_args, data_args, table_manager, library, vector_store, metrics) -> None:

    conn = table_manager.connect_to_database()
    prompt_manager = PromptManager(data_args, table_manager, library)

    tools = create_tools(conn, vector_store)
    agents = create_agents(model, data_args, tools)

    # load the fine-tuned model and ollama model
    # check if the fine-tuned model is provided
    if hasattr(model_args, "fine_tuned_model_path"):
        fine_tuned_model, tokenizer, _, device = load_models(model_args)
        is_use_fine_tuned_model_loaded = True
    else:
        # load bart_large model
        model_path = "facebook/bart-large"
        tokenizer = TapexTokenizer.from_pretrained(model_path, add_prefix_space=True)
        fine_tuned_model = BartForConditionalGeneration.from_pretrained(model_path)
        is_use_fine_tuned_model_loaded = False

    for i in range(data_args.num_iterations):
        try: 
            entries = run_pipeline_step(prompt_manager, 
                                    agents,
                                    tools, 
                                    fine_tuned_model,
                                    tokenizer,
                                    is_use_fine_tuned_model_loaded
                                    )
            if 'reason' in entries:
                print(f"Skipping example: {entries['reason']}")
                for i in range(data_args.num_iterations):
                    # concat the reason and the question and the answer
                    reason = entries['reason']
                    question = entries['question']
                    answer = entries['answer']
                    reason = f"Reason: {reason}\nQuestion: {question}\nAnswer: {answer}"
                    entries = run_pipeline_step(prompt_manager, 
                                        agents,
                                        tools, 
                                        fine_tuned_model,
                                        tokenizer,
                                        is_use_fine_tuned_model_loaded, 
                                        reason_previous_skip=reason
                                        )
                    if 'reason' not in entries:
                        break
                    else:
                        print(f"Skipping example: {entries['reason']}")
                        continue
            else:
                break
        except Exception as e:
            print(f"\033[91mPipeline step failed: {e}\033[0m")
            continue
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
            # print in red
            print(f"\033[91mFailed to generate entry, continuing...\033[0m")
    
    print(f"Dataset generation complete. Final library size: {len(library)}")
    del tools
    del agents
    del prompt_manager
    del conn
    gc.collect()

def main():
    # add time
    start_time = time.time()
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    table_manager = TableManager(data_args)
    model = load_model(model_args)

    metrics = {
        "retriever_too_similar_count": 0,
        "sql_empty_result_count": 0,
        "sql_execution_error_count": 0
    }

        
    print(f"chunk {data_args.chunk}/{data_args.Nchunks}")
    print(model_args)
    print(data_args)
    if model_args.fine_tuned_model_path is not None:
        model_args.model_name_or_path = model_args.fine_tuned_model_path
        print(f"Using fine-tuned model from {model_args.model_name_or_path}")
    else:
        print(f"Using bart-large model from {model_args.model_name_or_path}")
        
    if data_args.chunk is not None:
        data_args.library_path = data_args.library_path.split('.json')[0]+f"_chunk{data_args.chunk}_{data_args.Nchunks}.json"
        print(f'modification of the library path for chunks : {data_args.library_path}')
        print(f'previous common_ids length: {len(table_manager.common_ids)}')

        chunk_size = len(table_manager.common_ids) // data_args.Nchunks
        table_manager.common_ids = table_manager.common_ids[data_args.chunk*chunk_size:(data_args.chunk*chunk_size)+chunk_size]

        print(f"new common_ids length for chunk {data_args.chunk}: {len(table_manager.common_ids)}")
    library, vector_store = init_library(data_args)


    print(f"Starting with library containing {len(library)} entries")

    start_time = time.time()

    total = len(table_manager.common_ids)
    for idx, table_id in enumerate(table_manager.common_ids):
        
        new_start_time = time.time()

        table_manager.current_table_id = table_id
        try:
            generate_dataset(model, model_args, data_args, table_manager, library, vector_store, metrics)
        except FileNotFoundError as e:
            print(f"\033[91mSkipping table {table_id}: {str(e)}\033[0m")
            continue
        except Exception as e:
            print(f"\033[91mError processing table {table_id}: {str(e)}\033[0m")
            continue

        end_time = time.time()
        duration = end_time - new_start_time
        total_duration = end_time - start_time
        print(f"step {idx}/{total} done in {duration:.2f} seconds, total {total_duration:.2f} seconds.\n")



    os.makedirs("../data", exist_ok=True)
    with open("../data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time: {duration:.2f} seconds")
    # print in hours, minutes, seconds
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}:{int(minutes)}:{int(seconds)}")    


if __name__ == "__main__":
    main()


