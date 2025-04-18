import json
import os


def filter_function(merged_library):
    # List of substrings to flag as bad queries (case-insensitive)
    BAD_STRINGS = [
        "error:",
        "'E', 'r', 'r', 'o', 'r'",
        "simplification by hiding",
        "sematic substitution questio",
        "complete reformulation questio",
        "simplification with more",
        "paraphrase with change of perspective:"
    ]

    def is_bad_query(sql):
        sql_lower = sql.lower()
        return any(bad_str.lower() in sql_lower for bad_str in BAD_STRINGS)

    
    with open(merged_library, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    removed_count = 0

    for item in data:
        if is_bad_query(item["question"]):
            print(f"Removed question: {item['question']}")
            removed_count += 1
        else:
            filtered_data.append(item)

    os.remove(merged_library)


    with open(merged_library, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)

    # Print summary
    print(f"Total entries: {len(data)}")
    print(f"Removed entries: {removed_count}")
    print(f"Saved entries: {len(filtered_data)}")


    return merged_library