import json
from tqdm import tqdm
from transformers import HfArgumentParser, BartTokenizer
from datasets import Dataset, load_dataset, DatasetDict

# Replace these imports with your actual module paths
from source.utils.args import ModelArguments, DataArguments, TrainingArguments
from source.library.tables import TableManager

from source.prepare_data.filter_errors import filter_function
from source.prepare_data.merged_library import merged_function
from source.prepare_data.sql_executor import SQLExecutor


def is_bad_answer(data_args, tokenizer, answers):
    joined = ", ".join(answers).strip()
    joined_lower = joined.lower()

    # Rule 1: Detect SQL error messages
    if "error" in joined_lower or "execution failed" in joined_lower or "syntax error" in joined_lower:
        return True

    # Rule 2: Tokenized answer too long
    input_ids = tokenizer.encode(joined, truncation=False)
    return len(input_ids) >= data_args.max_target_length

def main():
    # Step 1: Load training args and initialize TableManager
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    table_manager = TableManager(data_args)

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

        try:
            # Get table
            if table_id != table_manager.current_table_id:
                table_manager.current_table_id = table_id
                conn = table_manager.connect_to_database()
                execute_sql= SQLExecutor(conn)  

            df = table_manager.get_durty_table()

            table_dict = {
                "header": list(df.columns),
                "rows": df.values.tolist()
            }

            # Get answers using SQL executor
            answers = execute_sql.forward(sql_query)
            answers = [str(cell) for row in answers for cell in row if cell is not None]
            print("answers: ", answers)

            if is_bad_answer(data_args, tokenizer, answers):
                print(f"[Filtered] Skipping bad answers: {answers}")
                bad_answer_count += 1
                continue
            # Append formatted data
            dataset_entries.append({
                "table": table_dict,
                "question": question,
                "answers": answers
            })

            

        except Exception as e:
            print(f"[Warning] Skipping table_id {table_id}: {e}")
            continue
        
    # Step 4: Convert to Hugging Face Dataset
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