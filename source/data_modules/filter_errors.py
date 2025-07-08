import json
import os
from source.utils.logger import setup_logger
logger = setup_logger(__name__)

def filter_function(library_path):
    # List of substrings to flag as bad queries (case-insensitive)
    BAD_STRINGS = [
        "error:",
        "'E', 'r', 'r', 'o', 'r'",
        "simplification by hiding",
        "sematic substitution questio",
        "complete reformulation questio",
        "simplification with more",
        "using synonyms:",
        "complete reformulation:",
        "simplifying by hiding details:",
        "semantic substitution:",
        "paraphrase with change of perspective:",
        "basic simplification:",
        "c1_number",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "```py",
        
    ]

    def is_bad_query(query):
        query_lower = query.lower().strip()

        if query_lower == "":
            return True
        
        return any(bad_str.lower() in query_lower for bad_str in BAD_STRINGS)

    
    with open(library_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    removed_count = 0

    for item in data:
        print(item["question"])
        if is_bad_query(item["question"]):
            logger.warning("Removed question: %s", item['question'])
            removed_count += 1
        else:
            filtered_data.append(item)

    os.remove(library_path)


    with open(library_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)

    # Print summary
    logger.info("Total entries: %d", len(data))
    logger.info("Removed entries: %d", removed_count)
    logger.info("Saved entries: %d", len(filtered_data))


    return library_path