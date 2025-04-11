# 🔬 LLM Code Smell Evaluation Experiment

This experiment evaluates how well various large language models (LLMs) detect **code smells** in code snippets using different prompt styles and input formats.

---

## 🎯 Goals

- Measure the accuracy of LLMs in identifying known code smells.
- Compare performance across:
  - Models (e.g., Mixtral, LLaMA, Phi)
  - Prompt types (`short`, `full`)
  - Code input styles (`diff_only`, `full_file_plus_diff`)
- Visualize results:
  - False Positives / False Negatives
  - ROC curves per model and prompt type

---

## 🧪 Experimental Setup

### Input Data

Each row in the input CSV contains:

- `full_file`: Original file contents
- `file_diff`: Code diff to review
- `expected_smells`: Known smells (e.g., `async`, `duplicate_code`, etc.)
- `language`: Either `dotnet` or `angular`

### Prompt Variants

- **Short Prompt**: Brief instruction for identifying code smells.
- **Full Prompt**: Detailed enterprise-grade guidelines for best practices.

### Code Input Types

- `diff_only`: Only the code diff is provided.
- `full_file_plus_diff`: Full file context + diff are combined.

---

## 🧠 Models

Evaluated across multiple OpenAI-compatible LLMs, such as:

- `mixtral-8x7b-instruct-v01`
- *(others available but commented out for now)*

---

## 📈 Metrics Collected

For each combination of model + prompt type + code input type:

- ✅ True Positives
- ❌ False Positives
- 🚫 False Negatives
- ✅ True Negatives (inferred)

**ROC curves** are plotted by treating each code smell as a binary classification.

---

## 📊 Visualizations

- **Bar Chart**: Summarizes false positives and false negatives.
- **ROC Curve**: Compares detection performance across models and prompt types.

---

## ⚙️ Running the Experiment

Install requirements:

```bash
pip install -r requirements.txt


## 🏆 Results Summary

| Rank | Model                     | ROC AUC |
|------|---------------------------|---------|
| 🥇   | `llama-3-1-8b-instruct`    | **0.89** |
| 🥈   | `llama-3-8b-instruct`      | 0.78    |
| 🥉   | `llama-3-sqlcoder-8b`      | 0.73    |

These scores represent the **average area under the ROC curve (AUC)** for code smell detection across all prompts and input types.
The scores were best with more detailed prompts with specific issues rather than a short general instruction.

![image](https://github.com/user-attachments/assets/db6851c9-c30e-4671-bd85-518413e99d13)

![image](https://github.com/user-attachments/assets/9b946e0b-130f-4541-8469-90010ee5a296)

![image](https://github.com/user-attachments/assets/c8221582-b8ee-43ec-af6c-9a8eaee74f2d)




---

*LLM code review is surprisingly strong! Future experiments may explore prompt tuning, fine-tuning, or hybrid approaches.*


