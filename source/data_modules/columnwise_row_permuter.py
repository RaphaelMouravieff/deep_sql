import os
import numpy as np
import sqlite3
import random
import pandas as pd
import logging
from source.data_modules.sql_executor import SQLExecutor


def mapping_cols(dirty_header, clean_header):
    """
    Map each dirty column to its corresponding cX columns in the clean header.

    Returns:
        Dict[str, List[str]] where keys are dirty column names, and values are list of clean col names.
    """
    mapping = {}
    # Only keep columns like c1, c1_x, c2, c2_y, etc.
    clean_columns = [col for col in clean_header if col.startswith("c")]

    # Group clean columns by prefix (c1, c2, ...)
    from collections import defaultdict
    grouped = defaultdict(list)
    for col in clean_columns:
        prefix = col.split("_")[0]  # e.g., 'c1'
        grouped[prefix].append(col)

    # Assign dirty columns by position to groups like c1, c2, ...
    for i, dirty_col in enumerate(dirty_header):
        group_key = f"c{i+1}"
        if group_key in grouped:
            mapping[dirty_col] = grouped[group_key]

    return mapping


def permute_columns_by_sql(clean_df, dirty_df, col_map, n_rows, base_seed):
    """
    Permute each column independently (row-wise), same permutation for each dirty/clean pair.
    """
    for i, (dirty_col, clean_cols) in enumerate(col_map.items()):
        np.random.seed(base_seed + i)
        permuted_indices = np.random.permutation(n_rows)

        for clean_col in clean_cols:
            clean_df.loc[:n_rows - 1, clean_col] = clean_df.loc[:n_rows - 1, clean_col].iloc[permuted_indices].values

        dirty_df.loc[:n_rows - 1, dirty_col] = dirty_df.loc[:n_rows - 1, dirty_col].iloc[permuted_indices].values

    return clean_df, dirty_df


def bad_answer_checker(data_args, tokenizer, answers):

    length = len(answers)
    if length > data_args.max_target_length :
        return True, answers

    answers = [str(cell) for row in answers for cell in row if cell is not None]
    if length > data_args.max_target_length:
        return True, answers

    if length == 0:
        return True, answers

    if length == 1:
        if answers[0] == "None" or answers[0] == "null":
            return True, answers

    joined = ", ".join(answers).strip().lower()

    if "error" in joined or "execution failed" in joined or "syntax error" in joined or "e, r, r, o, r" in joined:
        return True, answers

    if data_args.length_filter:
        input_ids = tokenizer.encode(joined, truncation=False)
        return len(input_ids) >= data_args.max_target_length, answers

    return False, answers


def permute_dirty_columns(dirty_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    col_indices = np.random.permutation(len(dirty_df.columns))
    permuted_columns = dirty_df.columns[col_indices]
    return dirty_df[permuted_columns]

def generate_sqlaware_permuted_examples(
    table_id: str,
    sql_query: str,
    question: str,
    original_conn,
    dirty_table: dict,
    executor_class,
    tokenizer,
    data_args,
    output_dir: str = "../data/tables/db_permuted",
    n: int = 10,
    base_seed: int = 42,
    counter:int = 0,
):
    """
    Generate SQL-aware column-wise row-permuted examples.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    clean_df = pd.read_sql_query("SELECT * FROM w", original_conn)
    dirty_df = pd.DataFrame(dirty_table["rows"], columns=dirty_table["header"])

    if clean_df.shape[0] != dirty_df.shape[0]:
        raise ValueError("Mismatch between clean and dirty row counts.")

    header = clean_df.columns.tolist()
    N = clean_df.shape[0]

    cols = dirty_df.columns.tolist()
    col_map = mapping_cols(cols, header)

    #if counter==0:
    #    print(f"\n[Original Clean Table: {table_id}]\n{clean_df.head(5)}\n")
    #    print(f"[Original Dirty Table]\n{dirty_df.head(5)}\n")

    for i in range(n):
        print(f"--- Permutation {i+1} ---")

        clean_df_copy = clean_df.copy()
        dirty_df_copy = dirty_df.copy()

        clean_df_copy, dirty_df_copy = permute_columns_by_sql(
            clean_df_copy, dirty_df_copy, col_map, N, base_seed + i * 100
        )
        dirty_df_copy = permute_dirty_columns(dirty_df_copy, base_seed + i * 100)
        if counter==0:
            #print(f"[Permuted Clean Table #{i+1}]\n{clean_df_copy.head(5)}\n")
            print(f"[Original Dirty Table]\n{dirty_df.head(5)}\n")
            print(f"[Permuted Dirty Table #{i+1}]\n{dirty_df_copy.head(5)}\n")



        db_path = os.path.join(output_dir, f"{table_id}_perm_{i}.db")
        new_conn = sqlite3.connect(db_path)
        clean_df_copy.to_sql("w", new_conn, index=False, if_exists="replace")
        new_conn.commit()

        try:
            executor = executor_class(new_conn)
            answers = executor.forward(sql_query)
            is_bad, answers = bad_answer_checker(data_args, tokenizer, answers)
            new_conn.close()

            if is_bad:
                print("[Filtered] Bad answer — skipping.\n")
                continue

            print(f"\n[Processing] {question}")
            print(f"[SQL Answer #{i+1}]: {answers}\n")
            print('\n' * 3)

            results.append({
                "table": {
                    "header": list(dirty_df_copy.columns),
                    "rows": dirty_df_copy.values.tolist()
                },
                "question": question,
                "answers": answers
            })

        except Exception as e:
            print(f"[Error in permutation {i+1}]: {e}")
            new_conn.close()
            os.remove(db_path)
            continue

    return results