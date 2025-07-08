

from datasets import load_from_disk


path = "/home/raphael.gervillie/deep_sql/data/training_dataset/step1"

data = load_from_disk(path)
print(data)

data["train"] = data["train"].filter(lambda example: example["question"].strip() != "")

print("After removing empty questions:", len(data["train"]))
print(data)


data.save_to_disk("/home/raphael.gervillie/deep_sql/data/training_dataset/step1_filtered")