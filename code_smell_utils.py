# code_smell_utils.py

import re

DOTNET_SMELLS = [
    "duplicate_code", "null_check", "deep_nesting", "async", "exception",
    "structural_duplication", "unused_code", "hardcoded_localhost", "commented_code"
]

ANGULAR_SMELLS = [
    "duplicate_code", "deep_nesting", "unsubscribed_observable", "state_management_violation",
    "structural_duplication", "unused_code"
]

def extract_smells_from_response(text, language):
    known_smells = DOTNET_SMELLS if language.lower() == "dotnet" else ANGULAR_SMELLS
    found = set()
    for smell in known_smells:
        if re.search(rf'\b{re.escape(smell)}\b', text, re.IGNORECASE):
            found.add(smell)
    return list(found)
