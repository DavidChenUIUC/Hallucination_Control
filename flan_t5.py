import argparse, torch, wandb, nltk, evaluate, os
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
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


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

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether to resume from checkpoint",
    )

    parser.add_argument(
        "--storing_name",
        type=str,
        default=None,
        help="Specify the storing name",
    )    

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Specify the learning rate",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Specify the batch size",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="Specify the lr scheduler",
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Specify the warmup ratio",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Specify the random seed",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="rouge",
        help="Specify the metric for evaluation. Options: rouge, bertscore",
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default="0.0",
        help="Specify the weight decay param",
    )

    args = parser.parse_args()
    return args

class FlatT5():
    def __init__(self):        
        self.args = parse_args()
        
        # Init WandB
        if self.args.eval_only: os.environ['WANDB_SILENT'] = 'true' ## Keep silent for evaluation 
        wandb.init(project="Flan_T5")
        
        # Set random seed
        self.set_rand_seed()
        
        # Load tokenizer of FLAN-t5-XL
        self.tokenizer_model=self.args.tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)

        self.metric = evaluate.load(self.args.metric)

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

        # Check if path exists, save datasets to disk if not for later easy loading
        if not os.path.exists("data/train"):
            self.tokenized_dataset["train"].save_to_disk("data/train")
        if not os.path.exists("data/eval"):
            self.tokenized_dataset["test"].save_to_disk("data/eval")

        # Setup trainer
        if self.args.eval_only:
            if self.args.resume_from_checkpoint is None:
                self.output_dir = 'eval_'+self.args.base_model
            else:
                self.output_dir= 'eval_'+self.args.resume_from_checkpoint.split("/")[-1]
        else:
            if self.args.storing_name is None and not self.args.resume_from_checkpoint:
                self.output_dir= self.args.base_model.split("/")[-1]+"_lr_"+str(self.args.lr)+"_bs_"+str(self.args.bs)+"_warmup_"+str(self.args.warmup_ratio)+"_seed_"+str(self.args.seed)
            elif self.args.resume_from_checkpoint:
                self.output_dir= "resume_"+self.args.resume_from_checkpoint.split("/")[-1]+"_"+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            else:
                self.output_dir=self.args.storing_name
        print(f"|- Set the output dir to {self.output_dir}")

        if self.args.resume_from_checkpoint is not None:
            self.setup_resume_model()
        else:
            self.setup_base_model()
            
        self.setup_trainer()

    def set_rand_seed(self):
        # Set a random seed value
        seed_value = self.args.seed

        # Set the random seed for numpy
        np.random.seed(seed_value)

        # Set the random seed for PyTorch on CPU
        torch.manual_seed(seed_value)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
            # limiting certain aspects of CUDA functionality, like this:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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
        try:
            if self.args.metric == 'bertscore':
                result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            else:
                result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
        except:
            assert False and "Check if the metric is in the option list in args parser if no other errors"
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        if self.args.metric == 'bertscore':
            # print(result)
            del result['hashcode']
            new_res = {}
            for k, v_list in result.items():
                new_res[k]= round(np.mean(v_list),4)
            return new_res
        else:
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
    
    def setup_base_model(self):
        # huggingface hub model id
        base_model = self.args.base_model
        
        print(f"|- Using model: {base_model}")

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
    
    def setup_resume_model(self):
        peft_model_id = self.args.resume_from_checkpoint
        config = PeftConfig.from_pretrained(peft_model_id)

        # load base LLM model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # Load the Lora model
        self.model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
    def setup_trainer(self):            
        # Define training args
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            auto_find_batch_size=True,
            learning_rate=self.args.lr,  # Initial learning rate
            num_train_epochs=5,  
            per_device_train_batch_size=self.args.bs,  
            per_device_eval_batch_size=self.args.bs,  
            logging_dir=f"{self.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="epoch",
            report_to="wandb",
            do_eval=True,
            evaluation_strategy='epoch',
            lr_scheduler_type=self.args.lr_scheduler,
            weight_decay=self.args.weight_decay,  # Include weight decay for regularization
            warmup_ratio=self.args.warmup_ratio,  # Warmup for the first 10% of training
        )

        # pprint(self.training_args.to_dict()) ## pretty print training args

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
        
        if self.args.resume_from_checkpoint is not None:
            print(f"|- *** Evaluating '{self.args.resume_from_checkpoint} ' ***")
        else:
            print(f"|- *** Evaluating '{self.args.base_model} ' ***")

        metrics = self.trainer.evaluate(metric_key_prefix="eval")
        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            self.trainer.log_metrics("eval", metrics)
        except:
            print("|- Failed to show log")
            
        try:
            self.trainer.save_metrics("eval", metrics)
            print(f"|- Metrics are saved to {self.output_dir}")
        except:
            print("|- Failed to save metrics")

if __name__ == "__main__":
    flat_t5 = FlatT5()
    flat_t5.run()