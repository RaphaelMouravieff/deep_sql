import json
from tqdm import tqdm
from transformers import HfArgumentParser, BartTokenizer, Seq2SeqTrainingArguments
from datasets import Dataset, load_dataset, DatasetDict

# Replace these imports with your actual module paths
from source.utils.args import ModelArguments, DataArguments
from source.library.tables import TableManager

from source.data_modules.filter_errors import filter_function
from source.data_modules.sql_executor import SQLExecutor

from source.data_modules.columnwise_row_permuter import generate_sqlaware_permuted_examples

import os
from source.utils.logger import setup_logger
logger = setup_logger(__name__)


def main():

    # Step 1: Load training args and initialize TableManager
    parser = HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    table_manager = TableManager(data_args)

    library_path = filter_function(data_args.library_path)

    tokenizer = BartTokenizer.from_pretrained("../models/bart-large")
    bad_answer_count = 0
    
    # Step 2: Load merged examples
    with open(library_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Step 3: Process each entry
    dataset_entries = []



    counter = 0
    for item in tqdm(raw_data, desc="Processing examples"):
        
        table_id = item["tables_id"]
        question = item["question"]
        sql_query = item["sql"]


        try:
            if table_id != table_manager.current_table_id:
                table_manager.current_table_id = table_id
                table_manager.connect_to_database()

            entries = generate_sqlaware_permuted_examples(
                table_id=table_manager.current_table_id,
                sql_query=sql_query,
                question=question,
                original_conn=table_manager.conn,
                dirty_table=table_manager.wtq_table_by_id[table_manager.current_table_id],
                executor_class=SQLExecutor,
                tokenizer=tokenizer,
                data_args=data_args,
                n=10,
                counter=counter
            )
            counter += 1
            dataset_entries.extend(entries)

        except Exception as e:
            logger.error("[Error] Skipping table %s due to: %s", table_id, repr(e))
            continue


    train_dataset = Dataset.from_dict({
        "table": [e["table"] for e in dataset_entries],
        "question": [e["question"] for e in dataset_entries],
        "answers": [e["answers"] for e in dataset_entries],
    })

    wtq = load_dataset("wikitablequestions")
    val_dataset = wtq["validation"]
    test_dataset = wtq["test"]

    # Step 4: Combine into a DatasetDict
    full_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })

    # Step 5: Save to disk
    full_dataset.save_to_disk(data_args.save_dataset_path)

    logger.info("\n--- Summary ---")
    logger.info("Total examples processed: %d", len(raw_data))
    logger.info("Examples with bad answers filtered: %d", bad_answer_count)
    logger.info("Final dataset size: %d", len(dataset_entries))


if __name__ == "__main__":
    main()