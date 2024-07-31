import json
import pandas as pd

json_name = 'GT_rationale_dataset.json'
df = pd.read_csv('./V_COT_output/GT/GT_rationale_dataset.csv')


# Dictionary to store the JSON data
data = {}

for index, row in df.iterrows():  # Note the slicing to select rows
    data[row['name']] = {
        "Question": row['question'],
        "GT_option": row['answer_option'],
        "GT_value": row['answer_value'],
        "GT_with_Rationale": row['answer_with_rationale']
    }

# Save the JSON data to a file
with open(json_name, "w") as json_file:
    json.dump(data, json_file, indent=4)  # Use indentation for better readability

print(f"JSON file '{json_name}' created successfully!")