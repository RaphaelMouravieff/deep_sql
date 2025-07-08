import os
import json

# Path to the JSON file
file_path = "data/library/library_step1/library.json"

# Print file size in bytes
file_size = os.path.getsize(file_path)
print(f"File size: {file_size} bytes")

# Load JSON and print 'question' field from each observation
with open(file_path, 'r') as f:
    data = json.load(f)



for i, obs in enumerate(data):
    question = obs.get('question')
    print(f"{i+1}. {question}")

    if question.strip() == "":
        print(f"Empty question found at index {i+1}")       
        print(f"Observation: {obs}")