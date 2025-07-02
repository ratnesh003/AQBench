## Repo Structure
### Datasets:
- `preprocessed/` - Contains pickle file(.pkl) of the NCAP funding data and state-level demographic information for faster python loading.
- `aqi_downloader.ipynb` — A Jupyter notebook to download air quality data directly from the CPCB website for a specified date range. 
- `questions.csv` — Contains 10,034 benchmark questions and corresponding metadata (category, question ID, etc.). 

### Scripts:
- `batch_generation.py` - Script for batch-processing large numbers of queries across models using transformers library. The changes in this file can be done according to LLM’s library.
- `eval_pipeline.py` - The main evaluation harness which loads the questions and model generated outputs, runs the code in a sandboxed environment, and calculates metrics such as exec@1 and pass@k.
- `code_eval_utils.py` - Contains internal functions used by the evaluation pipeline such as exception tracking and pass@k calculation.
- `run.sh` - Runs the `batch_generation.py` using `nohup`, based on model names.