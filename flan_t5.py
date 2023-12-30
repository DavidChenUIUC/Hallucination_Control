import argparse, torch, wandb, nltk, evaluate
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from pprint import pprint
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default="google/flan-t5-xxl",
        help="Specify which self.tokenizer to use",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="philschmid/flan-t5-xxl-sharded-fp16",
        help="Specify which model to use",
    )
    parser.add_argument(
        "--eval_only",
        type=bool,
        default=False,
        help="Whether to do eval only",
    )
    
    args = parser.parse_args()
    return args

def set_rand_seed():
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

class FlatT5():
    def __init__(self):
        # Set random seed
        set_rand_seed()
        
        # Init WandB
        wandb.init(project="Flan_T5")

        self.args = parse_args()

        # Load tokenizer of FLAN-t5-XL
        self.tokenizer_model=self.args.tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)

        self.metric = evaluate.load("rouge")

        # Load dataset from the hub
        self.dataset = load_dataset("samsum")
        print(f"Train dataset size: {len(self.dataset['train'])}")
        print(f"Test dataset size: {len(self.dataset['test'])}")
        # Train dataset size: 14732, Test dataset size: 819
        
        # Set seq length
        self.max_source_length = -1
        self.max_target_length = -1
        self.setup_seq_length()

        # Tokenize dataset
        self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
        print(f"Keys of tokenized self.dataset: {list(self.tokenized_dataset['train'].features)}")

        # Save datasets to disk for later easy loading
        self.tokenized_dataset["train"].save_to_disk("data/train")
        self.tokenized_dataset["test"].save_to_disk("data/eval")

        # Setup trainer
        self.output_dir="lora-flan-t5-xxl-1e-3_linear_warm0.1_seed1234"
        self.setup_model()
        self.setup_trainer()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=-1)  
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Note that other metrics may not have a `use_aggregator` parameter
        # and thus will return a list, computing a metric for each sentence.
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}   
    
    def setup_seq_length(self):
        # The maximum total input sequence length after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded.
        tokenized_inputs = concatenate_datasets([self.dataset["train"], self.dataset["test"]]).map(lambda x: self.tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
        input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
        # take 85 percentile of max length for better utilization
        self.max_source_length = int(np.percentile(input_lenghts, 85))
        print(f"Max source length: {self.max_source_length}")

        # The maximum total sequence length for target text after tokenization. 
        # Sequences longer than this will be truncated, sequences shorter will be padded."
        tokenized_targets = concatenate_datasets([self.dataset["train"], self.dataset["test"]]).map(lambda x: self.tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
        target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
        # take 90 percentile of max length for better utilization
        self.max_target_length = int(np.percentile(target_lenghts, 90))
        print(f"Max target length: {self.max_target_length}")
    
    def preprocess_function(self, sample,padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=sample["summary"], max_length=self.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all self.tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def setup_model(self):
        # huggingface hub model id
        base_model = self.args.base_model

        # load model from the hub
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")

        # Define LoRA Config 
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)

        # add LoRA adaptor
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

        # trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817

    def setup_trainer(self):
        if self.args.eval_only:
            self.output_dir = 'test_'+self.output_dir
        # Define training args
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,  # Initial learning rate
            num_train_epochs=5,  # Fixed as requested
            per_device_train_batch_size=32,  # Fixed as requested
            per_device_eval_batch_size=32,  # Fixed as requested
            logging_dir=f"{self.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="steps",
            report_to="wandb",
            lr_scheduler_type='linear',  # Linear scheduler
            warmup_ratio=0.1,  # Warmup for the first 10% of training
        )

        pprint(self.training_args.to_dict()) ## pretty print training args

        wandb.log(self.training_args.to_dict())

        # Ignore tokenizer pad token in the loss
        label_pad_token_id = -100
        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )

        # Create Trainer instance
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            compute_metrics = self.compute_metrics,
        )
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    def run(self):
        # train model
        if not self.args.eval_only:
            self.trainer.train()

            # Save LoRA model & tokenizer results
            peft_model_id="results_"+self.output_dir
            self.trainer.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
        else:
            print("|- *** Evaluate ***")

            metrics = self.trainer.evaluate(metric_key_prefix="eval")
            # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    flat_t5 = FlatT5()
    flat_t5.run()