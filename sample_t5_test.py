import torch
import numpy as np
import random
from random import randrange
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from tqdm import tqdm
from selfcheckgpt.modeling_mqag import MQAG
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from collections import defaultdict

# Set the seed for reproducibility
seed_value = 123
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)

# Load peft config for pre-trained checkpoint etc.
# peft_model_id = "./backup_ckpt/lora-flan-t5-xxl-1e-3_seed1234/checkpoint-2000"
# peft_model_id = "./backup_ckpt/results_lora-flan-t5-xxl-5e-4_adamw_seed1234"
peft_model_id = "./philschmid/flan-t5-xxl-sharded-fp16"
config = PeftConfig.from_pretrained(peft_model_id)

# Load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
model.eval()

print("Peft model loaded")
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mqag_model = MQAG(
    g1_model_type='race', # race (more abstractive), squad (more extractive)
    device=device
)
selfcheck_nli = SelfCheckNLI(device=device) # set device to 'cuda' if GPU is available
# Load dataset from the hub and get a sample
dataset = load_dataset("samsum")

eval_results = defaultdict(lambda: defaultdict(list))

cnt = 0
for i in tqdm(range(len(dataset["test"])), desc="Evaluating"):
    cnt+=1
    if cnt<15:
        continue
    sample = dataset['test'][i]
    sample_id = sample['id']
    # sample = dataset['test'][randrange(len(dataset["test"]))]

    # Prepare inputs and labels
    max_length = 50  # Define a consistent max_length for both input and labels
    input_ids = tokenizer(sample["dialogue"], return_tensors="pt", max_length=max_length, padding='max_length', truncation=True).input_ids.cuda()
    labels = tokenizer(sample["summary"], return_tensors="pt", max_length=max_length, padding='max_length', truncation=True).input_ids.cuda()

    # Generate outputs using the model
    outputs = model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    


    # Print sample input, prediction, and gold summary
    print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")
    print(f"Generated summary:\n{decoded_preds[0]}")
    print(f"Reference summary:\n{decoded_labels[0]}")
    continue

    # Metrics calculation

    ################
    ## BERT SCORE ##
    ################
    # print('---'* 20)
    bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    del bertscore_result['hashcode']
    tmp={}
    for k, v_list in bertscore_result.items():
        tmp["BertScore "+k] = np.mean(v_list)
    bertscore_result = tmp

    ################
    ##   rouge    ##
    ################
    # print('---'* 20)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)

    ################
    ##    MQAG    ##
    ################
    # print('---'* 20)
    print(np.shape(decoded_preds[0]))
    print(decoded_preds)
    print(decoded_preds[0])
    print('-'*10)
    print(decoded_labels)
    print(decoded_labels[0])
    # try:
    #     mqag_score = mqag_model.score(candidate=decoded_preds[0], reference=decoded_labels[0], num_questions=3, verbose=False)
    # except:
    #     print('retry')
    #     try:
    #         mqag_score = mqag_model.score(candidate=decoded_preds[0], reference=decoded_labels[0], num_questions=3, verbose=False)
    #     except:
    #         continue
    # # print(score['total_variation'])
    # tmp={}
    # for k, v in mqag_score.items():
    #     tmp["MQAG "+k] = v
    # mqag_score = tmp

#     ###################
#     ## SelfCheck-NLI ##
#     ###################
#     # print('---'* 20)
#     # Generate outputs using the model
#     # sample
#     sample1 = model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()
#     sample2 = model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()
#     sample3 = model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()

#     # Decode predictions and labels
#     sample1 = tokenizer.batch_decode(sample1, skip_special_tokens=True)[0]
#     sample2 = tokenizer.batch_decode(sample2, skip_special_tokens=True)[0]
#     sample3 = tokenizer.batch_decode(sample3, skip_special_tokens=True)[0]

#     sent_scores_nli = selfcheck_nli.predict(
#         sentences = decoded_preds[0],                          # list of sentences
#         sampled_passages = [sample1, sample2, sample3], # list of sampled passages
#     )
#     # print(sent_scores_nli)
#     num_nli_contr = 0
#     nli_threshold = 0.5397 ## https://github.com/potsawee/selfcheckgpt/issues/17
#     for n in sent_scores_nli:
#         if n < nli_threshold:
#             num_nli_contr+=1
#     nli_score = {"NLI Score": np.mean(sent_scores_nli), "NLI contradiction %": float(num_nli_contr/len(sent_scores_nli))}

#     print('\n'+'---'* 20)

    # Aggregate results
    for k, v in rouge_result.items():
        eval_results[sample_id]["Rouge " + k].append(v)
    for k, v in bertscore_result.items():
        eval_results[sample_id][k].append(v)
    # for k, v in mqag_score.items():
    #     eval_results[sample_id][k].append(v)
    # eval_results[sample_id]["NLI Score"].append(nli_score["NLI Score"])
    # eval_results[sample_id]["NLI contradiction %"].append(nli_score["NLI contradiction %"])

# Write results to a CSV file
with open('evaluation_results.csv', 'w', newline='') as csvfile:
    # Dynamically extract fieldnames from the collected eval_results
    fieldnames = ['sample_id'] + list(next(iter(eval_results.values())).keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for sample_id, metrics in eval_results.items():
        row = {'sample_id': sample_id}
        for metric, values in metrics.items():
            row[metric] = np.mean(values)  # Or use another appropriate method of aggregation
        writer.writerow(row)

print("Finished writing to evaluation_results.csv")