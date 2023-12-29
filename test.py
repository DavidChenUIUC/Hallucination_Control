import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
import evaluate
from transformers import EvalPrediction


# Initialize the Accelerator and Load the ROUGE metric
accelerator = Accelerator()
metric = evaluate.load("rouge")

# Load the trained model and tokenizer
model_path = "./results/checkpoint-2600"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Prepare the device and model
model, tokenizer = accelerator.prepare(model, tokenizer)

def tokenization(example):
    # Tokenize the dialogue
    tokenized_dialogue = tokenizer(example["dialogue"], padding="max_length", truncation=True, max_length=512)
    
    # Tokenize the summary (ground truth)
    tokenized_summary = tokenizer(example["summary"], padding="max_length", truncation=True, max_length=512)
    
    return {
        "inputs": tokenized_dialogue["input_ids"],
        "attention_mask": tokenized_dialogue["attention_mask"],
        "labels": tokenized_summary["input_ids"]  # Using input_ids of summary as labels
    }


# Load the testing dataset and prepare it
test_dataset = load_dataset("samsum", split="train")
test_dataset = test_dataset.map(tokenization, batched=True)

# Filter out only the necessary fields for the DataLoader
test_dataset.set_format(type='torch', columns=['inputs', 'attention_mask', 'labels'])

# Define a simple data collator that just converts lists to tensors
def simple_collator(features):
    return {key: torch.stack([f[key] for f in features]) for key in features[0]}

# Initialize the DataLoader with the custom collator
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=simple_collator)

# Define the function to generate responses
def generate_responses(batch):
    inputs = batch['inputs'].to(accelerator.device)
    attention_mask = batch['attention_mask'].to(accelerator.device)
    outputs = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=512)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Collect predictions and actual labels
all_preds = []
all_labels = []
all_pairs=[]
cnt= 0

def compute_metrics(predictions, labels):
    # Compute ROUGE metric
    rouge_results = metric.compute(predictions=predictions, references=labels)

    # Check if the results are given as confidence intervals or raw scores
    # and extract relevant ROUGE scores accordingly
    if isinstance(rouge_results["rouge1"], float):
        # If the scores are raw floats, use them directly
        rouge_scores = {key: value * 100 for key, value in rouge_results.items()}
    else:
        # If the scores are objects with confidence intervals, extract the mid value
        rouge_scores = {key: value.mid.fmeasure * 100 for key, value in rouge_results.items()}

    return rouge_scores


# Collect predictions and actual decoded labels
for batch in tqdm(test_dataloader, desc="Evaluating"):
    generated_summaries = generate_responses(batch)
    all_preds.extend(generated_summaries)
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in batch['labels'].tolist()]
    all_labels.extend(decoded_labels)
    if cnt > 5:  # For debugging, adjust as needed
        break
    cnt += 1

# Compute and print the metrics
results = compute_metrics(all_preds, all_labels)
print("Evaluation results:", results)
