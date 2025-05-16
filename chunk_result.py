from evaluate import load
from glob import glob
import pandas as pd
import numpy as np
import textwrap
import argparse
import logging
import json
import os 

# ! Accepting Model name from terminal

parser = argparse.ArgumentParser(description="Running this model result generation for eval questions.")
parser.add_argument("--model", type=str, required=True, help="The name of Model on Huggingface")
parser.add_argument("--starts", type=int, required=True, help="The number of response for starting result creation")
parser.add_argument("--ends", type=int, required=True, help="The number of response for ending result creation")
args = parser.parse_args()


model_name = args.model


# ! Assinging Constants 

SAMPLE_SIZE = 10
base_filepath = os.getcwd()

# ! Configuring Logs 

log_filename = f"{base_filepath}/chunk_result_logs.txt"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message):
    print(message, flush=True)
    logging.info(message)

log_message("Pipeline started.")

df = pd.read_csv('questions.csv')
starts = args.starts
ends = args.ends
df = df.loc[starts - 1 : ends - 1]

categories = df["category"].unique().tolist()

model_responses = []

for i in range(1, 500 + 1):
    for _, file in enumerate(glob(f'{base_filepath}/responses/{model_name}/*/{i}/*')):
        with open(file, 'r') as f:
            data = json.load(f)
            model_responses.append(data['generated_samples'].copy())

df['generated_samples'] = model_responses

code_eval = load("code_eval")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pass_at_K_on_df(n, df):
    results = []
    for _, row in df.iterrows():

        id = row["id"]
        answer = row["answer"]
        question = row["question"]
        category = row["category"]
        code = row["code"]
        sample = row["generated_samples"]

        refer = textwrap.dedent(f"""
        import pandas as pd
        import numpy as np
        df = pd.read_pickle("{base_filepath}/preprocessed/main_data.pkl")
        ncap_funding_df = pd.read_pickle("{base_filepath}/preprocessed/ncap_funding_data.pkl")
        states_df = pd.read_pickle("{base_filepath}/preprocessed/states_data.pkl")
        assert str(get_response(df,states_df,ncap_funding_df)).strip() == str({repr(answer)}).strip()
        """)

        question_folder = f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/{id}"
        question_file_path = f"{question_folder}/result.json"
        os.makedirs(question_folder, exist_ok=True)
        
        if os.path.isfile(question_file_path):
            log_message(f"Skipping question {id} as result already exists.")
            existing_results = pd.read_json(question_file_path).to_dict(orient="records")
            results.extend(existing_results)
            continue

        pass_at_k, result = code_eval.compute(
            references=[refer],
            predictions=[sample],
            timeout = 300,
            num_workers = 16,
            k=n,
        )

        question_results = []
        for i in range(SAMPLE_SIZE):
            res = {
                'id': id,
                'question': question,
                'answer': answer,
                'category': category,
                'model': model_name,
                'true_code': code,
                'pass@1': pass_at_k['pass@1'],
                'pass@2': pass_at_k['pass@2'],
                'pass@5': pass_at_k['pass@5'],
                'result': result[0][i][1]['result'],
                'status': result[0][i][1]['passed'],
                'sample': sample[i],
            }
            question_results.append(res)
            results.append(res.copy())
            log_message(res)

        pd.DataFrame(question_results).to_json(question_file_path, orient="records", indent=4)

    return results

# !

def save_results(df, file_path):
    result_data = pass_at_K_on_df([1, 2, 5], df)
    result_df = pd.DataFrame(result_data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result_df.to_json(file_path, orient="records", indent=4)
    return result_df

# ! 

aggregated_reults_df = save_results(df,f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/result.json")

# os.makedirs(os.path.dirname(f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/result.json"), exist_ok=True)
# aggregated_reults_df.to_json(f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/result.json", orient="records", indent=4)

# ! PLOT

result = aggregated_reults_df.drop(columns=["question", "answer", "true_code", "result", "status", "sample", "id", "model"]).groupby(["category"]).mean().reset_index()

import matplotlib.pyplot as plt

result = result.sort_values(by="pass@1", ascending=False)

pass_k_s = ["pass@1", "pass@2", "pass@5"]
num_categories = len(categories)
num_passs = len(pass_k_s)

x = np.arange(num_categories)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, pass_k in enumerate(pass_k_s):
    ax.bar(x + i * bar_width, result[pass_k], bar_width, label=pass_k)

ax.set_xticks(x + bar_width)
ax.set_xticklabels(categories, rotation=45, ha="right")

ax.set_xlabel("Category")
ax.set_ylabel("pass@k Values")
ax.set_title(f"pass@k's for Each Category of {model_name}")

ax.legend(title="Metrics")

save_path = f"{base_filepath}/chunk_charts/{model_name}/{starts}_{ends}_pass_k_results.png"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, bbox_inches="tight", dpi=300)

plt.show()

# ! PLOT

error_result = aggregated_reults_df.drop(columns=["question", "answer", "true_code", "result", "status", "sample", "id", "model"]).groupby(["category"]).sem().reset_index()

import numpy as np
import matplotlib.pyplot as plt

error_result = error_result.sort_values(by="pass@1", ascending=False)

pass_k_s = ["pass@1", "pass@2", "pass@5"]
num_categories = len(categories)
num_passs = len(pass_k_s)

x = np.arange(num_categories)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, pass_k in enumerate(pass_k_s):
    ax.bar(x + i * bar_width, error_result[pass_k], bar_width, label=pass_k)

ax.set_xticks(x + bar_width)
ax.set_xticklabels(categories, rotation=45, ha="right")

ax.set_xlabel("Category")
ax.set_ylabel("Error Values")
ax.set_title("Comparison of Errors for Each Category")

ax.legend(title="Metrics")

save_path = f"{base_filepath}/chunk_charts/{model_name}/{starts}_{ends}_error_results.png"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, bbox_inches="tight", dpi=300)

plt.show()