import pandas as pd
import openai
import re
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import httpx


CSV_PATH = "code_review_samples_smole.csv"


OPENAI_API_KEY = os.getenv("OPENAPITOKEN")  
OPEN_API_BASE_URL = os.getenv("OPENAPIBASEURL")


if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAIAPITOKEN environment variable not set")

if not OPEN_API_BASE_URL:
    raise EnvironmentError("OPENAPIBASEURL environment variable not set")


http_client = httpx.Client(verify=False)


openai.api_key = OPENAI_API_KEY
openai.base_url = OPEN_API_BASE_URL
openai.http_client = http_client


# Define experimental variables
MODELS = [
    "mixtral-8x7b-instruct-v01",
    # "mistral-7b-instruct-v03",
    # "llama-3-8b-instruct",
    # "llama-3-1-8b-instruct",
    # "llama-3-2-3b-instruct",
    # "phi-3-mini-128k-instruct",
    # "phi-3-5-moe-instruct",
    # "llama-3-3-70b-instruct",
    # "codellama-13b-instruct",
    # "llama-3-sqlcoder-8b"
]

CODE_INPUT_TYPES = ["diff_only", "full_file_plus_diff"]
PROMPT_TYPES = ["short", "full"]

DOTNET_SMELLS = [
    "duplicate_code", "null_check", "deep_nesting", "async", "exception",
    "structural_duplication", "unused_code", "hardcoded_localhost", "commented_code"
]

ANGULAR_SMELLS = [
    "duplicate_code", "deep_nesting", "unsubscribed_observable", "state_management_violation",
    "structural_duplication", "unused_code"
]


def build_prompt(row, code_input_type, prompt_type):
    short_prompt = "You are a code reviewer. Identify any code smells in the provided code. Return a list of issues."
    full_prompt = """
You are an enterprise code assistant. Carefully review the following code for any issues based on best practices:
- Avoid code duplication (DRY)
- Use null-conditional operators
- Minimize deep nesting
- Prefer async/await patterns
- Proper exception handling
- Remove commented-out or testing code
- Avoid hardcoded localhost or credentials
- In Angular, unsubscribe from observables to prevent memory leaks
- Follow proper state management practices using ngrx (avoid updating services directly)

Return any detected issues in a markdown table format.
"""
    system_prompt = short_prompt if prompt_type == 'short' else full_prompt
    user_content = row['full_file'] + "\n\n" + row['file_diff'] if code_input_type == 'full_file_plus_diff' else row['file_diff']
    return system_prompt, user_content


def extract_smells_from_response(text, language):
    table_matches = re.findall(r'\|\s*(.*?)\s*\|', text)
    known_smells = DOTNET_SMELLS if language.lower() == "dotnet" else ANGULAR_SMELLS
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
        expected_smells = [] if row['expected_smells'] == 'none' else row['expected_smells'].split(',')

        for model, code_input_type, prompt_type in itertools.product(MODELS, CODE_INPUT_TYPES, PROMPT_TYPES):
            system_prompt, user_content = build_prompt(row, code_input_type, prompt_type)
            try:
                response = openai.completions.create(
                    model=model,
                    prompt= system_prompt + "\n" + user_content,
                    max_tokens=2000,
                    temperature=0.1
                )
                reply = response.choices[0].text.strip()
                detected_smells = extract_smells_from_response(reply, row['language'])

                false_negatives = set(expected_smells) - set(detected_smells)
                false_positives = set(detected_smells) - set(expected_smells)

                results.append({
                    "id": row['id'],
                    "language": row['language'],
                    "model": model,
                    "code_input_type": code_input_type,
                    "prompt_type": prompt_type,
                    "false_negatives": len(false_negatives),
                    "false_positives": len(false_positives)
                })

            except Exception as e:
                print(f"Error with row {row['id']} using {model}, {code_input_type}, {prompt_type}: {e}")

    return pd.DataFrame(results)


def plot_results(df):
    grouped = df.groupby(['language', 'model', 'code_input_type', 'prompt_type']).sum()[['false_negatives', 'false_positives']]
    grouped.plot(kind='bar', figsize=(14, 7), title='False Positives / False Negatives by Configuration')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# === MAIN ===
# Uncomment to run:
results_df = evaluate()
plot_results(results_df)

