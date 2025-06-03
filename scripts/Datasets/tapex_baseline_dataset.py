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
    parser.add_argument("--save_baseline_dataset", type=str, required=True)
    args = parser.parse_args()

    step_dataset = load_from_disk(args.dataset_stepx_path)
    N = len(step_dataset["train"])
    print(f"Extracting {N} rows from train.json")

    processed = []
    with open("/home/raphael.gervillie/TabStruct/ptdata/train.json") as f:
        for i, line in enumerate(f):
        
            if i >= N:
                break
            item = json.loads(line)
            print(item)
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

    final_dataset.save_to_disk(args.save_baseline_dataset)
    print(f"Dataset saved to {args.save_baseline_dataset}")


if __name__ == "__main__":
    main()
