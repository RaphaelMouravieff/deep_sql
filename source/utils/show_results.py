import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import argparse
import json
import pandas as pd


from datasets import load_dataset
import json


from source.utils.training_utils import (
    get_denotation_accuracy,
   # get_sqa_denotation_accuracy,
    calculate_final_scores,
)   
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

import random

from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import random




def prediction_results_to_df(prediction_results):
    """
    Converts the prediction results dictionary into a pandas DataFrame,
    sorted by Robust Accuracy (%) in descending order.
    """
    rows = []
    for perturbation_type, metrics in prediction_results.items():
        row = {
            "Perturbation Type": perturbation_type,
            "Original Accuracy (%)": metrics["original_acc"],
            "Perturbed Accuracy (%)": metrics["perturbed_acc"],
            "Robust Accuracy (%)": metrics["robust_acc"],
            "Num Examples": metrics["num_examples"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Robust Accuracy (%)", ascending=False).reset_index(drop=True)
    return df



def merge_robust_accuracies(r0, r1, name0="tapex", name1="nous"):
    """
    Merges two robustness DataFrames, renames columns for clarity, 
    and includes difference in robustness.

    Returns:
        pd.DataFrame
    """
    r0_renamed = r0.rename(columns={
        "Robust Accuracy (%)": f"Robust Accuracy ({name0})"
    })
    r1_renamed = r1.rename(columns={
        "Robust Accuracy (%)": f"Robust Accuracy ({name1})"
    })

    merged = pd.merge(
        r0_renamed[["Perturbation Type", f"Robust Accuracy ({name0})", "Num Examples"]],
        r1_renamed[["Perturbation Type", f"Robust Accuracy ({name1})"]],
        on="Perturbation Type"
    )

    # Add difference column
    merged["Difference (nous - tapex)"] = (
        merged[f"Robust Accuracy ({name1})"] - merged[f"Robust Accuracy ({name0})"]
    ).round(2)

    # Reorder columns
    ordered = merged[[
        "Perturbation Type",
        f"Robust Accuracy ({name1})",
        f"Robust Accuracy ({name0})",
        "Num Examples",
        "Difference (nous - tapex)"
    ]]

    return ordered

def plot_robust_accuracy_comparison(df, model1="tapex", model2="nous", save_path="robust_accuracy_comparison.png"):
    labels = df["Perturbation Type"]
    x = np.arange(len(labels))
    width = 0.35

    acc1 = df[f"Robust Accuracy ({model1})"]
    acc2 = df[f"Robust Accuracy ({model2})"]
    diff = df["Difference (nous - tapex)"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, acc1, width, label=model1.capitalize())
    bars2 = ax.bar(x + width/2, acc2, width, label=model2.capitalize())

    # Annotate only significant differences
    for i, d in enumerate(diff):
        if abs(d) >= 3:
            ax.text(x[i], max(acc1[i], acc2[i]) + 1.5, f"{d:+.1f}%", ha='center', fontsize=8, color='red')

    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_title(f"Robustness Comparison: {model1.capitalize()} vs {model2.capitalize()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(60, 100)  # Final display range
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'savefig at {save_path}')
    plt.show()





def main(args):
    with open(args.tapex_path, "r") as f:
        output_tapex = json.load(f)

    with open(args.nous_path, "r") as f:
        output_nous = json.load(f)

    prediction_results = calculate_final_scores(output_tapex)
    r0 = prediction_results_to_df(prediction_results)
    print(r0)
    prediction_results = calculate_final_scores(output_nous)
    r1 = prediction_results_to_df(prediction_results)
    print(r1)
    r = merge_robust_accuracies(r0, r1)
    
    print(r)
    plot_robust_accuracy_comparison(r.iloc[1:].reset_index(drop=True), save_path=args.save_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare robust accuracy of TAPEX and NOUS outputs.")
    parser.add_argument("--tapex_path", type=str, required=True, help="Path to the TAPEX output JSON file")
    parser.add_argument("--nous_path", type=str, required=True, help="Path to the NOUS output JSON file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the NOUS output JSON file")

    args = parser.parse_args()
    main(args)