import torch
import evaluate
import numpy as np
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel

# Set a random seed value
seed_value = 1234
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    # Additionally, you might want to ensure that the behavior is deterministic by
    # limiting certain aspects of CUDA functionality, like this:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL = "LoftQ/Llama-2-7b-hf-4bit-64rank"
max_seq_length = 512
rouge_metric = evaluate.load("rouge")
train_dataset = load_dataset("samsum", split="train")
eval_dataset = load_dataset("samsum", split="validation")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL,
    subfolder="loftq_init",
    is_trainable=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


output_dir = "./results_llama_lr1e-4_scheduler_const"
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


trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = compute_metrics,
        # peft_config=peft_config,
        dataset_text_field="dialogue",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")