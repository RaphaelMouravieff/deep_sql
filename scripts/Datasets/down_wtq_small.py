
from datasets import load_dataset, DatasetDict


# Step 2: Load the dataset from Hugging Face
dataset = load_dataset("wikitablequestions")

# Step 3: Keep full train, and only 100 samples from validation and test
trimmed_dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["validation"].select(range(100)),
    "test": dataset["test"].select(range(100))
})

# Step 4: Save the dataset to disk
save_path = "/home/raphael.gervillie/deep_sql/data/fine_tuning/wikitablequestions_small"
trimmed_dataset.save_to_disk(save_path)

print(f"Trimmed dataset saved to: {save_path}")