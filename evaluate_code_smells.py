# %% [markdown]
# ## Load Required Libraries
# We'll use pandas, matplotlib, openai, etc.
# Created using jupytext evaluate_code_smells.py --to notebook

# %%
import pandas as pd
import openai
import os
import matplotlib.pyplot as plt
from code_smell_utils import ANGULAR_SMELLS, DOTNET_SMELLS, extract_smells_from_response
from collections import defaultdict
import itertools
import httpx
from sklearn.metrics import roc_curve, auc



# %% [markdown]
# ## Environment variables 
# For security these are not added to the source code

# %%
CSV_PATH = "code_review_samples_smole.csv"


OPENAI_API_KEY = os.getenv("OPENAPITOKEN")  
OPEN_API_BASE_URL = os.getenv("OPENAPIBASEURL")


if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAIAPITOKEN environment variable not set")

if not OPEN_API_BASE_URL:
    raise EnvironmentError("OPENAPIBASEURL environment variable not set")




# %% [markdown]
# ## Define OpenAI API 
# Server has a self-signed cert, for testing only, need to point the openAPI client to it and avoid errors 

# %%
http_client = httpx.Client(verify=False)
openai.api_key = OPENAI_API_KEY
openai.base_url = OPEN_API_BASE_URL
openai.http_client = http_client



# %% [markdown]
# ## Define available and relevant models for testing
# 
# Define experimental variables
# Best Run was llama-3-1-8b-instruct .89
# Second Best Run was llama-3-8b-instruct .78
# Third Place was was llama-3-sqlcoder-8b .73

MODELS = [
    # "mixtral-8x7b-instruct-v01",
    # "mistral-7b-instruct-v03",
    # "llama-3-8b-instruct",
     "llama-3-1-8b-instruct",
    # "llama-3-2-3b-instruct",
    # "phi-3-mini-128k-instruct",
    # "phi-3-5-moe-instruct",
    # "llama-3-3-70b-instruct",
    # "codellama-13b-instruct",
    # "llama-3-sqlcoder-8b"
]

CODE_INPUT_TYPES = ["diff_only", "full_file_plus_diff"]
PROMPT_TYPES = ["short", "full"]



# %% [markdown]
# ## Define Prompts
# Short general prompt and more well defined long prompt

# %%
def build_prompt(row, code_input_type, prompt_type):
    short_prompt = """
You are a code reviewer. Identify any code smells in the provided code.

Return the list of issues as a markdown table with two columns:
| Smell | Description |

Use the following smell types when applicable: async, exception, duplicate_code, null_check, deep_nesting, structural_duplication, unused_code, hardcoded_localhost, commented_code.

If no smells are found, return an empty markdown table.
"""
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

Return the results as a markdown table with two columns:
| Smell | Description |

Use one of the following predefined smell types in the Smell column:
duplicate_code, null_check, deep_nesting, async, exception, structural_duplication, unused_code, hardcoded_localhost, commented_code, unsubscribed_observable, state_management_violation.

Use each smell type only if it clearly applies. If there are no issues, return an empty table.
"""
    system_prompt = short_prompt if prompt_type == 'short' else full_prompt
    
    full_file = str(row.get("full_file", "") or "")
    file_diff = str(row.get("file_diff", "") or "")

    if code_input_type == 'full_file_plus_diff':
        user_content = full_file + "\n\n" + file_diff
    else:
        user_content = file_diff
    
    return system_prompt, user_content



# %% [markdown]
# ## Experiment Definition
# Call the LLM to determine if there is a code smell or not under various conditions: short prompt vs long prompt
# Code without any issues, with issues, both Angular and C# DotNet 

# %%
def evaluate():
    df = pd.read_csv(CSV_PATH)
    results = []

    for _, row in df.iterrows():
        expected_smells = [] if row['expected_smells'] == 'none' else row['expected_smells'].split(',')

        for model, code_input_type, prompt_type in itertools.product(MODELS, CODE_INPUT_TYPES, PROMPT_TYPES):
            system_prompt, user_content = build_prompt(row, code_input_type, prompt_type)
            try:
                print(f"row {row['id']} using {model}, {prompt_type}")
                response = openai.completions.create(
                    model=model,
                    prompt= system_prompt + "\n" + user_content,
                    max_tokens=2000,
                    temperature=0.1
                )
                reply = response.choices[0].text.strip()
                detected_smells = extract_smells_from_response(reply, row['language'])

                ALL_SMELLS = list(set(DOTNET_SMELLS + ANGULAR_SMELLS))

                true_positives = set(expected_smells) & set(detected_smells)
                false_negatives = set(expected_smells) - set(detected_smells)
                false_positives = set(detected_smells) - set(expected_smells)
                true_negatives = set(ALL_SMELLS) - (true_positives | false_positives | false_negatives)

                results.append({
                    "id": row['id'],
                    "language": row['language'],
                    "model": model,
                    "code_input_type": code_input_type,
                    "prompt_type": prompt_type,
                    "true_positives": len(true_positives),
                    "true_negatives": len(true_negatives),
                    "false_positives": len(false_positives),
                    "false_negatives": len(false_negatives),
                    "expected": expected_smells,
                    "detected": detected_smells
                })

            except Exception as e:
                print(f"Error with row {row['id']} using {model}, {code_input_type}, {prompt_type}: {e}")

    return pd.DataFrame(results)



# %% [markdown]
# ## ROC Plot Definition
# Plot ROC - Receiver Operating Characteristic 
# Values closer to 1 means more true positives less false positives

# %%
def plot_roc_by_model_and_prompt(df):
    models = df['model'].unique()
    prompt_types = df['prompt_type'].unique()
    all_smells = list(set().union(*df['expected']).union(*df['detected']))

    plt.figure(figsize=(12, 8))

    for model in models:
        for prompt_type in prompt_types:
            subset = df[(df['model'] == model) & (df['prompt_type'] == prompt_type)]
            y_true, y_score = [], []

            for _, row in subset.iterrows():
                expected = set(row["expected"])
                detected = set(row["detected"])
                for smell in all_smells:
                    y_true.append(1 if smell in expected else 0)
                    y_score.append(1 if smell in detected else 0)

            if any(y_true):  # only plot if there are positives
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                label = f"{model} ({prompt_type}) AUC={roc_auc:.2f}"
                plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Model and Prompt Type")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# %% [markdown]
# ## Run the program
# Run and plot the result
#

# %%
results_df = evaluate()
plot_roc_by_model_and_prompt(results_df)


