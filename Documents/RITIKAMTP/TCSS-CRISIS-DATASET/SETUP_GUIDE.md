# Crisis Summarization Project: Setup & Usage Guide

This guide explains how to run the crisis summarization project, including data preparation and environment setup.

## Prerequisites
- Python 3.8+
- GPU (recommended for running BigBird and BLIP-2 models)

## Installation

1.  **Clone the Repository**: (You already did this)
    ```bash
    git clone https://github.com/Raghvendra-14/A-Multimodal-Approach-and-Dataset-to-Crisis-Summarization-in-Tweets
    cd 'New folder\CrisisSummarization\Documents\RITIKAMTP\TCSS-CRISIS-DATASET'
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## Data Preparation

The provided code expects specific JSON files which are not part of the repository by default (`tweetsample.json` and `bbcsample.json`). I have created helper scripts to generate these from the available CSV data.

1.  **Generate `tweetsample.json`**:
    Run the provided script to convert `BangaloreRiots.csv` (or any other CSV) into the format required by the notebook:
    ```bash
    python convert_csv_to_json.py
    ```
    This script reads `BangaloreRiots.csv` and creates `tweetsample.json` containing a list of tweets.

2.  **Generate `summary_output.json`**:
    The multimodal part of `CODE_CRISIS_SUMM_TWEETS.ipynb` relies on an abstractive summary generated from news data. I have created a placeholder `summary_output.json` to allow the code to run even without news data.

## Running the Code

1.  **Tweet Summarization**:
    Open `CODE_CRISIS_SUMM_TWEETS.ipynb` in Jupyter Notebook or Google Colab.
    
    This notebook performs:
    *   **Preprocessing**: Cleans and tokenizes tweets.
    *   **Extractive Summarization**: Selects key tweets based on TF-IDF.
    *   **Multimodal Summarization**: Matches the summary text with images using CLIP, generates captions with BLIP-2, and synthesizes a final summary.

2.  **News Summarization** (Optional):
    If you have news data (e.g., `bbcsample.json`), you can run `CRISIS_SUMM_ON_NEWS.ipynb`.
    
## Troubleshooting
- **Missing Files**: Ensure `tweetsample.json` exists in the same directory as the notebook. Run `convert_csv_to_json.py` to recreate it.
- **Memory Issues**: The models (BigBird, BLIP-2) are large. If you encounter OOM errors, try reducing batch sizes or running on a machine with more VRAM (e.g., Google Colab A100/T4).
