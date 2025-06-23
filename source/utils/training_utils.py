
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional
import numpy as np


delimiter = ", "

def calculate_final_scores(output_data):
    orig_accs = {}
    for example in output_data:
        if example["perturbation_type"] == "original":
            orig_accs[example["original_id"]] = example["accuracy"]

    types = ["original", "synonym", "abbreviation", "row", "column", "extend", "masked", "add", "word", "sentence", "combined"]

    accs = {}
    for type in types:
        accs[type] = {
            "original": [],
            "perturbed": [],
            "both_correct": []
        }
        for example in output_data:
            if example["perturbation_type"] == type:
                accs[type]["perturbed"].append(example["accuracy"])
                accs[type]["original"].append(orig_accs[example["original_id"]])
                if example["accuracy"] == 1 and orig_accs[example["original_id"]] == 1:
                    accs[type]["both_correct"].append(1)

    prediction_results = {}
    for type in types:
        if accs[type]["original"] == []:
            continue
        prediction_results[type] = {
            "original_acc": round(np.mean(accs[type]["original"])*100, 4),
            "perturbed_acc": round(np.mean(accs[type]["perturbed"])*100, 4),
            "robust_acc": round(sum(accs[type]["both_correct"]) / sum(accs[type]["original"])*100, 1),
            "num_examples": len(accs[type]["original"]),
        }

    return prediction_results



def evaluate_example(predict_str: str, ground_str: str):
    delimiter = ", "
    predict_spans = predict_str.split(delimiter)
    ground_spans = ground_str.split(delimiter)
    predict_values = defaultdict(lambda: 0)
    ground_values = defaultdict(lambda: 0)
    for span in predict_spans:
        try:
            predict_values[float(span)] += 1
        except ValueError:
            predict_values[span.strip()] += 1
    for span in ground_spans:
        try:
            ground_values[float(span)] += 1
        except ValueError:
            ground_values[span.strip()] += 1
    _is_correct = predict_values == ground_values
    return _is_correct


def get_denotation_accuracy(predictions: List[str], references: List[str]):
    assert len(predictions) == len(references)
    correct_num = 0
    for predict_str, ground_str in zip(predictions, references):
        is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
        if is_correct:
            correct_num += 1
    return correct_num / len(predictions)

    