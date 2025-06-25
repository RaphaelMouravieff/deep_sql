import argparse
from datasets import load_dataset, DatasetDict
from source.utils.logger import setup_logger
logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Trim Hugging Face datasets and save to disk.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trimmed dataset")
    parser.add_argument("--train_size", type=int, default=-1, help="Number of train samples to keep (-1 for all)")
    parser.add_argument("--val_size", type=int, default=-1, help="Number of validation samples to keep")
    parser.add_argument("--test_size", type=int, default=-1, help="Number of test samples to keep")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the full dataset
    dataset = load_dataset("wikitablequestions")

    # Apply filtering based on args
    trimmed_dataset = DatasetDict()

    # Train
    if args.train_size > 0:
        trimmed_dataset["train"] = dataset["train"].select(range(args.train_size))
    else:
        trimmed_dataset["train"] = dataset["train"]

    # Validation
    if args.val_size > 0:
        trimmed_dataset["validation"] = dataset["validation"].select(range(args.val_size))
    else:
        trimmed_dataset["validation"] = dataset["validation"]

    # Test
    if args.test_size > 0:
        trimmed_dataset["test"] = dataset["test"].select(range(args.test_size))
    else:
        trimmed_dataset["test"] = dataset["test"]

    # Save
    logger.info("Trimmed dataset: %s", trimmed_dataset)
    trimmed_dataset.save_to_disk(args.save_path)
    logger.info("Trimmed dataset saved to: %s", args.save_path)

if __name__ == "__main__":
    main()