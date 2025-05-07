import json
from tqdm import tqdm
from transformers import HfArgumentParser, BartTokenizer, Seq2SeqTrainingArguments
from datasets import Dataset, load_dataset, DatasetDict

# Replace these imports with your actual module paths
from source.utils.args import ModelArguments, DataArguments
from source.library.tables import TableManager

from source.prepare_data.filter_errors import filter_function
from source.prepare_data.sql_executor import SQLExecutor

from source.prepare_data.merged_library import merged_function
from source.prepare_data.columnwise_row_permuter import generate_sqlaware_permuted_examples

import os



def main():

    # Step 1: Load training args and initialize TableManager
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    table_manager = TableManager(data_args)

    merged_path = os.path.join(data_args.merged_library_folder, "merged_library.json")
    if os.path.exists(merged_path):
        os.remove(merged_path)

    merged_library = merged_function(data_args.merged_library_folder)
    merged_library = filter_function(merged_library)

    tokenizer = BartTokenizer.from_pretrained("../models/bart-large")
    bad_answer_count = 0
    

    # Step 2: Load merged examples
    with open(merged_library, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Step 3: Process each entry
    dataset_entries = []




    for item in tqdm(raw_data, desc="Processing examples"):
        table_id = item["tables_id"]
        question = item["question"]
        sql_query = item["sql"]

        print(question)

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
            n=10
        )
        dataset_entries.extend(entries)


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

    print("\n--- Summary ---")
    print(f"Total examples processed: {len(raw_data)}")
    print(f"Examples with bad answers filtered: {bad_answer_count}")
    print(f"Final dataset size: {len(dataset_entries)}")

if __name__ == "__main__":
    main()