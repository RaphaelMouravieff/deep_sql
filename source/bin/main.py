
from tqdm import tqdm
from transformers import HfArgumentParser
import json
import os
import numpy as np

from source.prompts.prompt import  PromptManager
from source.library.tables import TableManager
from source.utils.args import  (ModelArguments,
                               DataArguments)

from source.bin.main_step import run_pipeline_step
from source.library.storage import  init_library, save_library, save_vector_store
from source.models.build_agents import create_agents
from source.tools.build_tools import create_tools
from source.models.llm_model_setup import load_model_llm

import gc
import time

from collections import defaultdict


from source.models.ft_model_setup import load_model_ft, load_tokenizer, load_config
from source.utils.logger import setup_logger
from source.tools.answer_checker import AnswerChecker


from source.utils.logger import setup_logger
logger = setup_logger(__name__)


def generate_dataset(model, data_args, table_manager, vector_store, answer_checker) -> None:


    conn = table_manager.connect_to_database()
    prompt_manager = PromptManager(data_args, table_manager, vector_store)

    tools = create_tools(conn, vector_store)
    agents = create_agents(model, data_args, tools)


    infos = defaultdict(int)
    logger.info("Initializing infos...")

    inside_error = 0
    for idx in range(data_args.num_iterations):


        entries, inside = run_pipeline_step(prompt_manager, 
                                            agents,
                                            answer_checker)
        


        if not inside[0]:
            inside_error +=1
            logger.info('Changing prompt entry due to inside condition: %s', inside)
            prompt_manager = PromptManager(data_args, table_manager, vector_store, inside)
            continue


        if entries is not None:

            entries[-1]["too_similar"] = tools["retriever_tool"].too_similar_count
            entries[-1]["empty_sql"] = tools["execute_sql"].empty_result_count
            entries[-1]["execution_error"] = tools["execute_sql"].execution_error_count
            entries[-1]["likelihood"] = inside[3]  # log-likelihood
            entries[-1]["num_iterations"] = idx + 1
            entries[-1]["inside_error"] = inside_error
            

            save_library(data_args, entries)
            save_vector_store(data_args, entries)
            break

        else:
            logger.info("Failed to generate entry, continuing... inside: %s", inside)

    logger.info("Dataset generation complete.")
    del tools
    del agents
    del prompt_manager
    del conn
    gc.collect()

def main():
    
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    table_manager = TableManager(data_args)
    model_llm = load_model_llm(model_args)


    logger.info("chunk %d/%d", data_args.chunk, data_args.Nchunks)



    if data_args.chunk is not None:        
        chunk_size = len(table_manager.common_ids) // data_args.Nchunks
        table_manager.common_ids = table_manager.common_ids[data_args.chunk*chunk_size:(data_args.chunk*chunk_size)+chunk_size]

        logger.info("new common_ids length for chunk %d: %d", data_args.chunk, len(table_manager.common_ids))

    
    answer_checker = None
    
    if model_args.use_model_check:

        with open("../logs/likelihood.json") as data:
            results = json.load(data)
        likelihood = [result["log_likelihood"] for result in results]
        accuracy = [result["accuracy"] for result in results]
        lower_thresh = np.percentile(likelihood, 5)
        upper_thresh = np.percentile(likelihood, 80)
        logger.info("Lower threshold: %s, Upper threshold: %s", lower_thresh, upper_thresh)
        
        config = load_config(model_args, logger)
        tokenizer = load_tokenizer(model_args, logger)
        model_ft = load_model_ft(model_args, config, logger).to("cuda")
        answer_checker = AnswerChecker( model=model_ft,
                                        tokenizer=tokenizer,
                                        data_args=data_args, 
                                        lower_thresh=lower_thresh, 
                                        upper_thresh=upper_thresh )





    start_time = time.time()
    total = len(table_manager.common_ids)

    for idx, table_id in enumerate(table_manager.common_ids):
        
        try:
            vector_store = init_library(data_args)
            new_start_time = time.time()
            table_manager.current_table_id = table_id

            generate_dataset(model_llm, data_args, table_manager, vector_store, answer_checker)

            end_time = time.time()
            duration = end_time - new_start_time
            total_duration = end_time - start_time
            logger.info("step %d/%d done in %.2f seconds, total %.2f seconds.", idx, total, duration, total_duration)

        except Exception as e:
            logger.error("Failed to load vector store for table_id %s: %s", table_id, str(e))
            continue  # skip to next table_id


if __name__ == "__main__":
    main()


