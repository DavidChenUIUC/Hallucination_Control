import torch
import numpy as np
import random
from random import randrange
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate, csv
from tqdm import tqdm
import time, pickle
from selfcheckgpt.modeling_mqag import MQAG
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from collections import defaultdict
from datetime import datetime
from qa_gpt import QA_GPT


class SummaryEvaluator:
    def __init__(self):
        self.overall_eval_results = defaultdict(list)
        self.THRESHOLD = 0.5
        self.RETRY_LIMIT = 3
        self.split = 'test'
        
        self.testing_num = 100
        # self.rand_list = [randrange(len(self.dataset)) for i in range(self.testing_num)]
        # self.dataset = load_dataset("samsum")[self.split]
        
        dataset = load_dataset("samsum")[self.split]
        self.dataset = defaultdict(lambda: defaultdict(str) )
        for sample in dataset:
            self.dataset[sample['id']] = sample

        self.rand_list = ['13682496', '13681246', '13819724', '13829029', '13680227', '13681603', '13730685', '13862536', '13819925', '13862663', '13716653']

        self.set_seed(123)
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")
        self.load_models_and_tokenizers()
        self.qa_gpt = QA_GPT()

        self.nli_threshold = 0.5397 
        self.num_nli_sample = 3

        self.save_metrics = False
        self.save_summaries_only = True
        self.file_name='hal_reduction_eval_results.csv'

    def set_seed(self, seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)

    def load_models_and_tokenizers(self):
        # Load peft config for pre-trained checkpoint etc.
        peft_model_id = "./backup_ckpt/lora-flan-t5-xxl-1e-3_seed1234/checkpoint-2000"
        # peft_model_id = "./backup_ckpt/results_lora-flan-t5-xxl-5e-4_adamw_seed1234"

        self.config = PeftConfig.from_pretrained(peft_model_id)

        # Load base LLM model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0})
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)

        # Load the Lora model
        self.model = PeftModel.from_pretrained(self.model, peft_model_id, device_map={"": 0})
        self.model.eval()

        print("Peft model loaded")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mqag_model = MQAG(g1_model_type='race', device=self.device)
        self.selfcheck_nli = SelfCheckNLI(device=self.device)

    def get_metrics(self, decoded_preds, decoded_labels, input_ids, labels):
        eval_results = defaultdict()
        
        # BERTScore
        bertscore_result = self.bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        del bertscore_result['hashcode']
        for k, v_list in bertscore_result.items():
            eval_results["BertScore " + k] = round(np.mean(v_list),4)

        # Rouge
        rouge_result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
        for k, v in rouge_result.items():
            eval_results["Rouge " + k] = round(v,4)
        
        # MQAG
        try:
            mqag_score = self.mqag_model.score(candidate=decoded_preds[0], reference=decoded_labels[0], num_questions=3, verbose=False)
        except Exception as e:
            print('MQAG scoring failed:', str(e))
            mqag_score =  {'kl_div': 0, 'counting': 0, 'hellinger': 0, 'total_variation': 0}
            # return None
        for k, v in mqag_score.items():
            eval_results["MQAG " + k] = round(v,4)

        # SelfCheck-NLI
        self.nli_threshold = 0.5397
        nli_samples = []
        for _ in range(self.num_nli_sample):
            nli_samples.append(self.generate_summary(input_ids, labels)[0][0]) # decoded_preds[0]
        sent_scores_nli = self.selfcheck_nli.predict(
            sentences=decoded_preds[0],  # list of sentences
            sampled_passages=nli_samples,  # list of sampled passages
        )
        num_nli_contr = sum(score < self.nli_threshold for score in sent_scores_nli)
        nli_score = {"NLI Score": round(np.mean(sent_scores_nli),4), "NLI contradiction %": round(float(num_nli_contr / len(sent_scores_nli)),4)}
        eval_results.update(nli_score)

        return eval_results

    def verify_summary(self, document, summary):
        correctness = 0
        questions, options, gpt_answers, gpt_qa_exec_time = self.qa_gpt.qa(document, summary)
        for i in range(len(questions)):
            probs = self.mqag_model.answer(questions=[{'question':questions[i], 'options':options[i]}], context=summary) #[{'question': question, 'options': options}]
            slm_answer = np.argmax(probs)
            if slm_answer == gpt_answers[i]:
                correctness += 1 / len(questions)
        return (correctness, questions, gpt_answers, gpt_qa_exec_time, (len(questions)==0))

    def generate_summary(self, input_ids, labels):
        outputs = self.model.generate(input_ids=input_ids, max_length=50, do_sample=True, top_p=1.0).detach().cpu().numpy()
        decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return decoded_preds, decoded_labels
    
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.random_forest = pickle.load(file)
        print(f"Model loaded from {filename}")

    def evaluate_summaries(self):
        for i in tqdm(self.rand_list, desc="Evaluating"):
            start = time.time()
            sample = self.dataset[i]
            sample_id = sample['id']
            # prefix = "The summrization "+sample["summary"]+" for the dialogue below is wrong. Please do it again."
            # input_ids = self.tokenizer(prefix+sample["dialogue"], return_tensors="pt", max_length=50, padding='max_length', truncation=True).input_ids.to(self.device)
            input_ids = self.tokenizer(sample["dialogue"], return_tensors="pt", max_length=50, padding='max_length', truncation=True).input_ids.to(self.device)
            labels = self.tokenizer(sample["summary"], return_tensors="pt", max_length=50, padding='max_length', truncation=True).input_ids.to(self.device)

            decoded_preds, decoded_labels = self.generate_summary(input_ids, labels)
            first_shot_summary = decoded_preds[0]
            correctness, questions, gpt_answers, gpt_qa_exec_time, no_questions = self.verify_summary(decoded_preds[0], decoded_labels[0])
            retry = 10
            while no_questions or (correctness < self.THRESHOLD and retry <= self.RETRY_LIMIT):
                print(f"** Retrying {retry} **")
                retry += 1
                decoded_preds, decoded_labels = self.generate_summary(input_ids, labels)
                correctness, questions, gpt_answers, gpt_qa_exec_time, no_questions = self.verify_summary(decoded_preds[0], decoded_labels[0])
            
            if no_questions:
                print("** No questions generated after retries **")
                continue
            elif (correctness < self.THRESHOLD and retry < self.RETRY_LIMIT):
                print("** Correctness below threshold after retries **")
        
            if self.save_metrics:
                metrics = self.get_metrics(decoded_preds, decoded_labels, input_ids, labels) # input_ids, labels are for generating additional samples for MQAG
                end = time.time()
                if metrics is not None:
                    metrics["id"] = sample_id
                    metrics["correctness"] = round(correctness,4)
                    metrics["num_questions"] = len(questions)
                    metrics["questions"] = questions
                    metrics["gpt_answers"] = gpt_answers
                    metrics["gpt_qa_exec_time"] = gpt_qa_exec_time
                    metrics["overall_exec_time"] = round((end-start),4)
                    metrics["tries"] = retry
                    try:
                        metrics['dialogue']=sample["dialogue"]
                        metrics['final_summary']=decoded_preds[0]
                        metrics['first_shot_summary']=first_shot_summary
                    except:
                        pass
                    self.overall_eval_results[sample_id] = metrics
                else:
                    print("|- Not adding current metric to result to due popped up error")
                    
            if self.save_summaries_only:
                self.file_name = 'no_prefix_hal_reducted_samsum_hal_summaries.csv'
                self.overall_eval_results[sample_id] = {'id': sample_id,\
                                                        'dialogue': sample["dialogue"],\
                                                        'final_summary': decoded_preds[0],\
                                                        'first_shot_summary': first_shot_summary,}

        self.save_results_to_csv()

    def save_results_to_csv(self):
        # Add time to file name
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = self.file_name.replace('.csv', f'_{cur_time}.csv')

        with open(file_name, 'w', newline='') as csvfile:
            fieldnames = ['id'] + list(next(iter(self.overall_eval_results.values())).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for sample_id, metrics in self.overall_eval_results.items():
                row = {'id': sample_id}
                for metric, values in metrics.items():
                    if isinstance(values, list) and isinstance(values[0], (int, float)):
                        row[metric] = np.mean(values) 
                    elif isinstance(values, list) and isinstance(values[0], str):
                        row[metric] = values
                    else:
                        row[metric] = values

                writer.writerow(row)
        print(f"Finished writing to {file_name}")

if __name__ == "__main__":
    # Create an instance of the class and call the evaluation method
    evaluator = SummaryEvaluator()
    evaluator.evaluate_summaries()
