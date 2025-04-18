import os
import json
from glob import glob

# Folder containing the JSON files
folder_path = '../data/library'
output_file = '../data/library/merged_library.json'

# Get all JSON file paths
json_files = glob(os.path.join(folder_path, '*.json'))

# Load and combine all data
merged_data = []
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        merged_data.extend(json.load(f))

# Write to a single JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2)

print('Merged JSON files into:', output_file)
print('Length of merged data:', len(merged_data))