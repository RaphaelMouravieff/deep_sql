

import json
import torch

from source.models.ft_model_setup import load_config, load_tokenizer, load_model_ft
from source.data_modules.data_loader import load_datasets
from source.utils.logger import setup_logger
from source.tools.answer_checker import AnswerChecker
from source.utils.metrics import evaluate_example
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy.stats import pearsonr

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
question = "how many movies does Leonardo Di Caprio have?"
expected_answer = ["53"]


class SimpleArgs:
    max_source_length: int = 1024
    max_target_length: int = 128
    pad_to_max_length: bool = True
    num_beams: int = 5
    use_model_check: bool = True   
    model_name_or_path: str = "models/bart_large_step0/checkpoint-10000"
    tokenizer_name: str = None
    config_name = None
    use_fast_tokenizer = True
    output_generation: bool = True
    dataset_name = "data/fine_tuning/wikitablequestions"
    
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = setup_logger()

datasets = load_datasets(SimpleArgs, logger)

config = load_config(SimpleArgs, logger)

tokenizer = load_tokenizer(SimpleArgs, logger)

model = load_model_ft(SimpleArgs, config, logger)

model = model.to("cuda")




if __name__ == "__main__":
    checker = AnswerChecker(model=model, tokenizer=tokenizer, data_args=SimpleArgs(), device=device)

    results = []

    size = len(datasets['validation'])
    for idx, example in enumerate(datasets['validation']):
        table = pd.DataFrame.from_records(example["table"]["rows"], columns=example["table"]["header"])
        question = example['question']
        expected_answer = example['answers']


        print(f'--- Example {idx}/{size} ---')
        print('Table:', table.head())
        print('Question:', question)
        

        model_answer, log_likelihood = checker.check_answer(table, question, expected_answer)
        accuracy = evaluate_example(model_answer.lower(), ", ".join(expected_answer).lower())
        print()
        print('Model answer:', model_answer)
        print('Expected answer:', expected_answer)  
        print('Log likelihood:', log_likelihood)
        print("accuracy", accuracy)
        print('\n'*3)

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "log_likelihood": log_likelihood,
            "accuracy": accuracy
        })

    likelihood = [result["log_likelihood"] for result in results]
    accuracy = [result["accuracy"] for result in results]

    correlation, p_value = pearsonr(likelihood, accuracy)
    print(f"Pearson correlation: {correlation:.4f} (p={p_value:.4e})")

    lower_thresh = np.percentile(likelihood, 10)  # bottom 10%
    upper_thresh = np.percentile(likelihood, 90)  # top 10%

    print(f"Lower threshold (complex): {lower_thresh}")
    print(f"Upper threshold (simple): {upper_thresh}")


    total_accuracy = sum(accuracy) / len(accuracy)
    print(f"Total Accuracy: {total_accuracy:.4f}")  

    with open("logs/likelihood.json", "w") as f:
        json.dump(results, f, indent=4)


    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 5), dpi=300)
    
    sns.histplot(
        likelihood,
        bins=40,
        kde=False,
        stat="density",
        color="lightgray",
        edgecolor="black",
        label="Histogram"
    )
    sns.kdeplot(
        likelihood,
        color="blue",
        linewidth=2,
        label="KDE"
    )
    plt.axvline(lower_thresh, color='red', linestyle='--', linewidth=1.5, label='10th percentile')
    plt.axvline(upper_thresh, color='green', linestyle='--', linewidth=1.5, label='90th percentile')
    plt.xlabel("Log-Likelihood")
    plt.ylabel("Density")
    plt.title("Distribution of Log-Likelihoods")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("logs/plots/log_likelihood_distribution.png", bbox_inches='tight', dpi=300)
    plt.show()

