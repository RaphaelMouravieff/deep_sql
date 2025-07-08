import argparse
import json
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk


def reconstruct(flattened: str) -> tuple:
    parts = flattened.split("||")    
    query = parts[0]  
    header = parts[1].split("|") 
    rows = [row.split("|") for row in parts[2:]] 
    table = {
        "header": header,
        "rows": rows
    }
    return table, query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_stepx_path", type=str, required=True)
    parser.add_argument("--actual_baseline_path", type=str, required=True)
    parser.add_argument("--previous_baseline_path", type=str, required=True)
    parser.add_argument("--full_pt_baseline_path",
                        type=str,
                        default="/home/raphael.gervillie/TabStruct/ptdata/train.json")
    
    args = parser.parse_args()

    step_dataset = load_from_disk(args.dataset_stepx_path)
    N = len(step_dataset["train"])
    print(f"Extracting {N} rows from train.json")

    previous_baseline_step_dataset = load_from_disk(args.previous_baseline_path)
    N_previous = len(previous_baseline_step_dataset["train"])
    print(f"Ignoring the {N_previous} rows from previous baseline dataset")


    processed = []
    with open(args.full_pt_baseline_path) as f:
        for i, line in enumerate(f):
        
            if i < N_previous:
                continue

            if i >= N+N_previous:
                break
            item = json.loads(line)
            table_dict, recon_query = reconstruct(item["table"])
            processed.append({
                "question": item["question"],
                "answers": item["answers"],
                "table": table_dict
            })
            print(f"Processed row {i}: question={item['question'][:30]}... table keys={list(table_dict.keys())}")

    train_dataset = Dataset.from_list(processed)

    wtq = load_dataset("wikitablequestions")
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": wtq["validation"],
        "test": wtq["test"]
    })

    final_dataset.save_to_disk(args.actual_baseline_path)
    print(f"Dataset saved to {args.actual_baseline_path}")
    print('Dataset structure:',final_dataset)

if __name__ == "__main__":
    main()
