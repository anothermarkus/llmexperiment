# llmexperiment
Need a good python script to run through code smell scenarios through llm.

I have used LLM to generate a bunch of CSV files to iterate through and test the LLMs for the effectiveness.

Currently this is not working, working through this error:

Error with row 2 using mixtral-8x7b-instruct-v01, diff_only, short: can only concatenate str (not "float") to str
Error with row 2 using mixtral-8x7b-instruct-v01, diff_only, full: can only concatenate str (not "float") to str
Traceback (most recent call last):
  File "c:\github.com\llmexperiment\evaluate_code_smells.py", line 141, in <module>
    results_df = evaluate()
  File "c:\github.com\llmexperiment\evaluate_code_smells.py", line 102, in evaluate
    system_prompt, user_content = build_prompt(row, code_input_type, prompt_type)
  File "c:\github.com\llmexperiment\evaluate_code_smells.py", line 79, in build_prompt
    user_content = row['full_file'] + "\n\n" + row['file_diff'] if code_input_type == 'full_file_plus_diff' else row['file_diff']
TypeError: can only concatenate str (not "float") to str



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/<your-notebook>.ipynb) <-- This link is also not working but looks cool
