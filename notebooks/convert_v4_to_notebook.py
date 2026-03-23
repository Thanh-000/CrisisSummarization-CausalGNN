"""Convert .py script to .ipynb notebook."""
import json
import re

INPUT = "causalcrisis_v4_llava_experiment.py"
OUTPUT = "CausalCrisis_V4_LLaVA_Experiment.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    content = f.read()

# Split on '# %%' markers
raw_cells = re.split(r'\n# %%\n', content)

cells = []

for i, cell_content in enumerate(raw_cells):
    cell_content = cell_content.strip()
    if not cell_content:
        continue
    
    source_lines = cell_content.split('\n')
    source_with_newlines = [line + '\n' for line in source_lines]
    # Remove trailing newline from last line
    if source_with_newlines:
        source_with_newlines[-1] = source_with_newlines[-1].rstrip('\n')
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_with_newlines
    })

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "A100"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Created %s with %d cells" % (OUTPUT, len(cells)))
