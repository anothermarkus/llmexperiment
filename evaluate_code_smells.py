import pandas as pd
import openai
import re
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# === CONFIGURATION ===
CSV_PATH = "/mnt/data/code_review_samples.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # expects API key to be set as environment variable

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

openai.api_key = OPENAI_API_KEY

def build_prompt(row):
    short_prompt = "You are a code reviewer. Identify any code smells in the provided code. Return a list of issues."
    full_prompt = """
You are an enterprise code assistant. Carefully review the following code for any issues based on best practices:
- Avoid code duplication (DRY)
- Use null-conditional operators
- Minimize deep nesting
- Prefer async/await patterns
- Proper exception handling

Return any detected issues in a markdown table format.
"""
    
    system_prompt = short_prompt if row['prompt_type'] == 'short' else full_prompt
    user_content = row['code_sample'] if row['code_input_type'] == 'full_file_plus_diff' else f"Code diff:\n{row['code_sample']}"
    
    return system_prompt, user_content

def extract_smells_from_response(text):
    table_matches = re.findall(r'\|\s*(.*?)\s*\|', text)
    # Crude filtering for keywords that look like code smells
    known_smells = [
        "duplicate_code", "null_check", "deep_nesting", "async", "exception", 
        "structural_duplication", "unused_code"
    ]
    found = set()
    for match in table_matches:
        for smell in known_smells:
            if smell.lower() in match.lower():
                found.add(smell)
    return list(found)

def evaluate():
    df = pd.read_csv(CSV_PATH)
    results = []

    for _, row in df.iterrows():
        system_prompt, user_content = build_prompt(row)

        try:
            response = openai.ChatCompletion.create(
                model=row['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            reply = response.choices[0].message.content
            detected_smells = extract_smells_from_response(reply)
            expected_smells = [] if row['expected_smells'] == 'none' else row['expected_smells'].split(',')

            false_negatives = set(expected_smells) - set(detected_smells)
            false_positives = set(detected_smells) - set(expected_smells)

            results.append({
                "id": row['id'],
                "language": row['language'],
                "model": row['model'],
                "false_negatives": len(false_negatives),
                "false_positives": len(false_positives)
            })

        except Exception as e:
            print(f"Error with row {row['id']}: {e}")

    return pd.DataFrame(results)

def plot_results(df):
    grouped = df.groupby(['language', 'model']).sum()[['false_negatives', 'false_positives']]
    grouped.plot(kind='bar', figsize=(10, 6), title='False Positives / False Negatives by Language & Model')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# === MAIN ===
# Uncomment to run:
# results_df = evaluate()
# plot_results(results_df)
