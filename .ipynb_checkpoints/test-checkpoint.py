import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
import evaluate

# Initialize the Accelerator and Load the ROUGE metric
accelerator = Accelerator()
metric = evaluate.load("rouge")

# Load the trained model and tokenizer
model_path = "./exp_results/test/step_0"
# model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Prepare the device and model
model, tokenizer = accelerator.prepare(model, tokenizer)

def tokenization(example):
    src_examples = [f"Your task is to write a summarization for the following dialogue: '{src}' " for src in example['dialogue']]
    tokenized_dialogue = tokenizer(src_examples, padding="max_length", truncation=True, max_length=256)

    trg_examples = [f"Summarization: '{src}' " for src in example['summary']]
    tokenized_summary = tokenizer(trg_examples, padding="max_length", truncation=True, max_length=256)

    return {
        "input_ids": tokenized_dialogue["input_ids"],
        "attention_mask": tokenized_dialogue["attention_mask"],
        "labels": tokenized_summary["input_ids"]  # Using input_ids of summary as labels
    }

# Load the testing dataset and prepare it
test_dataset = load_dataset("samsum", split="test")
test_dataset = test_dataset.map(tokenization, batched=True)

# Filter out only the necessary fields for the DataLoader
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define a simple data collator that pads inputs and labels to the maximum length in a batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the DataLoader with the custom collator
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

# Define the function to generate responses
def generate_responses(batch):
    inputs = batch['input_ids'].to(accelerator.device)
    attention_mask = batch['attention_mask'].to(accelerator.device)
    outputs = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=512)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Collect predictions and actual labels
all_preds = []
all_labels = []

# Collect predictions and actual decoded labels
for batch in tqdm(test_dataloader, desc="Evaluating"):
    generated_summaries = generate_responses(batch)
    all_preds.extend(generated_summaries)
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in batch['labels'].tolist()]
    all_labels.extend(decoded_labels)

# Compute and print the metrics
results = metric.compute(predictions=all_preds, references=all_labels)
print("Evaluation results:", results)
