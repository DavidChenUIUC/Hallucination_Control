import torch
import evaluate
import numpy as np
from trl import SFTTrainer
from datasets import load_dataset
from accelerate import Accelerator
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, IntervalStrategy

# Set a random seed value
seed_value = 1234

# Set the random seed for numpy
np.random.seed(seed_value)

# Set the random seed for PyTorch on CPU
torch.manual_seed(seed_value)

# If you are using PyTorch with CUDA (GPU), set the seed for all GPUs as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    # Additionally, you might want to ensure that the behavior is deterministic by
    # limiting certain aspects of CUDA functionality, like this:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
max_seq_length = 256
accelerator = Accelerator()
rouge_metric = evaluate.load("rouge")
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
train_dataset = load_dataset("samsum", split="train")
eval_dataset = load_dataset("samsum", split="validation")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

output_dir = "./results_lr2e-3_scheduler_cosine"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 1
learning_rate = 2e-3
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "cosine"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    # fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    # evaluation_strategy=IntervalStrategy.STEPS,  # or IntervalStrategy.EPOCH
    # eval_steps=300,
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    print(eval_preds)
    predictions = np.argmax(logits, axis=-1)  # This line is likely incorrect for causal language models
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-processing: we need to strip off the padding from the predictions and labels.
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]  # ROUGE expects a list of references for each prediction

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"rouge": result}


trainer = accelerator.prepare(\
    SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = compute_metrics,
        peft_config=peft_config,
        dataset_text_field="dialogue",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )\
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")