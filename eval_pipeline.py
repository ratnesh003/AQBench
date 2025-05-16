import argparse
import logging
import json

# ! Accepting Model name from terminal

parser = argparse.ArgumentParser(description="Running this model for eval questions.")
parser.add_argument("--model", type=str, required=True, help="The name of Model on Huggingface")
args = parser.parse_args()

model_name = args.model
print(f"RESPONSE GENERATION STARTED FOR MODEL : {model_name}")

# ! Loading model to GPU

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = torch.device('cuda:1')
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ! Assinging Constants 

SAMPLE_SIZE = 10

import os 
base_filepath = os.getcwd()

# ! Read system prompt from the default file

SYSTEM_PROMPT_FILE = "system_prompt.txt"

if not os.path.exists(SYSTEM_PROMPT_FILE):
    raise FileNotFoundError(f"System prompt file '{SYSTEM_PROMPT_FILE}' not found.")

with open(SYSTEM_PROMPT_FILE, "r") as prompt_file:
    system = prompt_file.read().strip()

# ! Configuring Logs 

log_filename = f"{base_filepath}/pipeline_logs.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message):
    print(message, flush=True)
    logging.info(message)
log_message("Pipeline started.")


# !


import pandas as pd
df = pd.read_csv("questions.csv")

categories = df["category"].unique().tolist()

# !

def post_process(code):
    ans = []
    for ins in code :
        if '<think>' in ins:
            parts = ins.split('</think>', 1)
            ins = parts[1] if len(parts) > 1 and parts[1].strip() else ""
        ins = ins.split('</code>')[0]
        ins = ins.replace('```python', '')
        ins = ins.split('```')[0]
        ins = ins.replace('<code>', '')
        ans.append(ins)
    return ans

# !

def querying_api(model_name, question, id, i):
    while True:
        try:
            chat = [
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ]
            formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False).to(device)
            outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True, temperature = 0.2, min_p = 0.1)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            log_message(f"Success : (Model : {model_name}, question : {id}, sample : {i})")
            break
        except Exception as e:
            log_message(f"Error : (Model : {model_name}, question : {id}, sample : {i})")
            print(e)

    return response

# !

def sample_responses(model_name, question, id, sample, path):
    
    generated_samples = []
    for i in range(sample):
        temp = querying_api(model_name, question, id, i)
        generated_samples.append(temp)

    data =  {
        "id": id,
        "generated_samples": post_process(generated_samples)
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w+") as f :
        json.dump(data, f, indent=4)

    return data

# !

import os

for category in categories:

    file_path = f"{base_filepath}/responses/{model_name}/{category}/response.json"

    if os.path.isfile(file_path) :
        log_message(f"Resopnse Generation : \nCategory : \"{category}\" completed for Model : \"{model_name}\"")
        continue

    temp_df = df[df["category"] == category]
    
    model_responses = []

    for idx, i in temp_df.iterrows():
        question = i["question"]
        id = i["id"]

        que_file_path = f"{base_filepath}/responses/{model_name}/{category}/{id}/response.json"

        if os.path.isfile(que_file_path) :
            with open(que_file_path, "r") as f:
                data = json.load(f)
                model_responses.append(data.copy())
            log_message(f"Resopnse Generation : \nCategory : \"{category}\" completed for Question : \"{id}\"")
            continue
        
        query_responses = sample_responses(model_name, question, id, sample=SAMPLE_SIZE, path=que_file_path)
        model_responses.append(query_responses.copy())

    response_df = pd.DataFrame(model_responses)
    save_df = pd.merge(temp_df, response_df, on="id", how="left", suffixes=("", ""))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    save_df.to_json(file_path, orient="records", indent=4)

# !

import textwrap
from evaluate import load
import os
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

        question_folder = f"{base_filepath}/results/{model_name}/{category}/{id}"
        question_file_path = f"{question_folder}/result.json"
        os.makedirs(question_folder, exist_ok=True)
        
        if os.path.isfile(question_file_path):
            log_message(f"Skipping question {id} as result already exists.")
            # Load existing results if already processed
            existing_results = pd.read_json(question_file_path).to_dict(orient="records")
            results.extend(existing_results)
            continue

        pass_at_k, result = code_eval.compute(
            references=[refer],
            predictions=[sample],
            timeout = 600,
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

def save_category_results(df, file_path):
    result_data = pass_at_K_on_df([1, 2, 5], df)
    result_df = pd.DataFrame(result_data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result_df.to_json(file_path, orient="records", indent=4)
    return result_df

# ! 

aggregated_reults_df = pd.DataFrame(columns=['id', 'question', 'answer', 'category', 'model', 'true_code', 'pass@1', 'pass@2', 'pass@5', 'result', 'status', 'sample'])

for category in categories:
    
    file_path = f"{base_filepath}/results/{model_name}/{category}/result.json"

    if os.path.isfile(file_path) :
        log_message(f"Result Generation : \nCategory : {category} completed for Model : {model_name}")
        category_result_df = pd.read_json(file_path)
        aggregated_reults_df = pd.concat([aggregated_reults_df, category_result_df])
        continue

    category_response_df = pd.read_json(f"{base_filepath}/responses/{model_name}/{category}/response.json")

    category_result_df = save_category_results(category_response_df, file_path)

    aggregated_reults_df = pd.concat([aggregated_reults_df, category_result_df])

os.makedirs(os.path.dirname(f"{base_filepath}/results/{model_name}/aggregated_result/result.json"), exist_ok=True)
aggregated_reults_df.to_json(f"{base_filepath}/results/{model_name}/aggregated_result/result.json", orient="records", indent=4)

# !

result = aggregated_reults_df.drop(columns=["question", "answer", "true_code", "result", "status", "sample", "id", "model"]).groupby(["category"]).mean().reset_index()

import numpy as np
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

save_path = f"{base_filepath}/charts/{model_name}_results.png"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, bbox_inches="tight", dpi=300)

plt.show()

# !

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

save_path = f"{base_filepath}/charts/{model_name}_error_results.png"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, bbox_inches="tight", dpi=300)

plt.show()
