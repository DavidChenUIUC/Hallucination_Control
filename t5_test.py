import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import PeftModel, PeftConfig
import evaluate

# Load peft config for pre-trained checkpoint etc. 
peft_model_id = "./lora-flan-t5-xxl-1e-3_seed1001/checkpoint-500"
config = PeftConfig.from_pretrained(peft_model_id)

# Load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load metric
metric = evaluate.load("rouge")

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")

# Load test dataset from disk and format it for PyTorch
test_dataset = load_from_disk("data/eval/")
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Initialize the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=32,
    do_predict=True,
    predict_with_generate=True
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=None  # We will compute metrics separately
)

# Run predictions
pred, label,_ = trainer.predict(test_dataset)

def compute_metrics(predictions, labels):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)  
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

res = compute_metrics(pred, label)
for k , v in res.items():
    print(f"{k}: {v}")
    
# # print results
# print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
# print(f"rouge2: {rogue['rouge2']* 100:2f}%")
# print(f"rougeL: {rogue['rougeL']* 100:2f}%")
# print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")
