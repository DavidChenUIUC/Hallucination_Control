from collections import defaultdict
import csv, torch, evaluate
import numpy as np
import pandas as pd
from pprint import pprint
from datasets import load_dataset  # Ensure 'datasets' library is installed.
import warnings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PegasusTokenizer, PegasusForConditionalGeneration
from datetime import datetime
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from selfcheckgpt.modeling_mqag import MQAG
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
warnings.filterwarnings("ignore")

class XSum_Hallucination:
    def __init__(self, file_path='./hallucination_annotations_xsum_summaries.csv'):
        """Initialize the class with the path to the CSV file and pre-load datasets."""
        self.file_path = file_path
        x_sum_dataset = load_dataset('xsum')['test']  # Pre-load the XSum dataset.
        self.x_sum = defaultdict(dict)
        for entry in x_sum_dataset:
            self.x_sum[entry['id']] = entry        
        
        self.systems = set()
        self.summary_annotations = defaultdict(list)
        self.combined_xsum = defaultdict(list)
        self.system_summaries = defaultdict(lambda: defaultdict(dict))
        
        self.make_data() 
        
        # Initialize NLI evaluation setup
        self.initialize_nli_evaluation()

    def make_data(self):
        """Load CSV file and merge it with the XSum dataset, storing the data in instance variables."""
        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for entry in reader:
                lookup_id = entry['bbcid']
                local_dict = {
                    'id': lookup_id,
                    'docs': self.x_sum[lookup_id]['document'],
                    'machine_summary': entry['summary'],
                    'system': entry['system'],
                    'hallucination_type': [entry['hallucination_type']],
                    'hallucinated_span': [entry['hallucinated_span']],
                    'worker_id': entry['worker_id']
                }
                self.systems.add(entry['system'])
                self.combined_xsum[lookup_id].append(local_dict)
                self.system_summaries[entry['system']][lookup_id] = local_dict
                self.summary_annotations[lookup_id].append((entry['system'], entry['hallucination_type']))
        
    def data_statistic(self):
        """Calculate and return statistics on the data."""
        systems_statistic = defaultdict(lambda: {"cnt": 0, "IorE_cnt": 0, "intrinsic": 0, "extrinsic": 0, "IorE": 0, "NULL": 0})

        for summary_id, annotations in self.summary_annotations.items():
            system_check = {k: {"INTRINSIC": False, "EXTRINSIC": False, "NO_NULL": True} for k in self.systems}
            for system, annotation in annotations:
                self.update_statistics(systems_statistic, system_check, system, annotation)

        return systems_statistic

    def update_statistics(self, stats, check, system, annotation):
        """Update the statistics based on the annotation."""
        stats[system]["cnt"] += 1
        if annotation == "intrinsic":
            stats[system]["intrinsic"] += 1
            check[system]["INTRINSIC"] = True
        elif annotation == "extrinsic":
            stats[system]["extrinsic"] += 1
            check[system]["EXTRINSIC"] = True
        elif annotation == "NULL":
            check[system]["NO_NULL"] = False

        if check[system]["NO_NULL"]:
            stats[system]["IorE"] += 1

    def print_statistic(self):
        """Print the calculated statistics."""
        print(f"Total summary count: {len(self.summary_annotations)}")
        systems_statistic = self.data_statistic()
        for system in self.systems:
            self.print_system_statistics(system, systems_statistic[system])

    def print_system_statistics(self, system, stats):
        """Print statistics for a specific system."""
        I_hallucination_rate = 100 * stats['intrinsic'] / max(stats['cnt'], 1)
        E_hallucination_rate = 100 * stats['extrinsic'] / max(stats['cnt'], 1)
        IorE_hallucination_rate = I_hallucination_rate + E_hallucination_rate
        faithfulness_rate = 100 - IorE_hallucination_rate
        print('-' * 40)
        print(f"--- System: {system}, Count: {stats['cnt']} ---")
        print(f"I-hallucination rate: {I_hallucination_rate:.2f} %")
        print(f"E-hallucination rate: {E_hallucination_rate:.2f} %")
        print(f"I or E hallucination rate: {IorE_hallucination_rate:.2f} %")
        print(f"Faithfulness rate: {faithfulness_rate:.2f} %")

    def save_statistics_to_excel(self, file_name='xsum-hal-system_statistics.xlsx'):
        """Save the statistics and summaries into an Excel file with each system in a separate tab."""
        try:
            # Create a Pandas Excel writer using openpyxl as the engine.
            with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                sheet_created = False  # Flag to track if at least one sheet has been created.

                # Iterate through each system to create a separate sheet.
                for system in self.systems:
                    # Collect data for the current system.
                    data = []
                    for doc_id, entry in self.system_summaries[system].items():
                        data.append([
                            doc_id,  # Document ID
                            system,  # System
                            entry['docs'],  # Document
                            entry['machine_summary'],  # Summary
                            ', '.join(entry['hallucination_type'])  # Hallucination Type
                        ])

                    # Only create a sheet if there's data.
                    if data:
                        # Convert the data into a pandas DataFrame.
                        df = pd.DataFrame(data, columns=["document_id", "system", "document", "summary", "hallucination_type"])
                        
                        # Write the DataFrame to a sheet in the Excel file.
                        df.to_excel(writer, sheet_name=system, index=False)
                        sheet_created = True

                if not sheet_created:
                    # Create a dummy sheet if no data was added to ensure file is created.
                    pd.DataFrame().to_excel(writer, sheet_name='No Data', index=False)

            if sheet_created:
                print(f"Statistics and summaries have been saved to {file_name}.")
            else:
                print(f"No data to save. An empty file was created with a 'No Data' sheet.")

        except Exception as e:
            print(f"An error occurred: {e}")

    def initialize_nli_evaluation(self):
        """Set up for NLI evaluation with necessary models and tokenizers."""
        # Set the seed for reproducibility
        seed_value = 123
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(self.device)
        self.model.eval()

        # Load additional models and metrics
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")
        self.mqag_model = MQAG(g1_model_type='race', device=self.device)
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
        
        self.storing_name = 'evaluation_results'
    def import_and_evaluate(self, npy_path):
        """Import the stored numpy array and evaluate using nli_evaluation."""
        # Import the stored numpy array
        id_system_arr = np.load(npy_path)
        id_system_set = set()
        for (bbcid, system) in id_system_arr:
            id_system_set.add((bbcid, system))
        print(f"NULL summaries set count: {len(id_system_set)}")
        
        # Initialize evaluation results dictionary
        eval_results = defaultdict(lambda: defaultdict(list))

        # Iterate through the imported tuples and perform nli_evaluation
        cnt = 0
        for bbcid, system in tqdm(id_system_set, total=len(id_system_set), desc='Evaluating'):
            # Retrieve the corresponding sample from the combined_xsum using bbcid
            sample = self.combined_xsum[bbcid]

            # Since sample might have multiple entries, we iterate through them
            for s in sample:
                # Check if the system matches
                if s['system'] == system:
                    cnt+=1
                    eval_results = self.nli_evaluation(s, eval_results, dataset=None)  # Assuming the 'dataset' parameter isn't used in nli_evaluation
                    if cnt==10 or cnt%100==0:
                        self.write_results_to_csv(eval_results)
                    break  # Break after evaluating the matching entry
        print(f"|- Runned on {cnt} data")
        # Consider implementing CSV writing in a separate method for flexibility.
        self.write_results_to_csv(eval_results)
        
    def nli_evaluation(self, sample, eval_results, dataset):
        eval_results_copy = eval_results.copy()
        """Perform NLI evaluation on the dataset."""
        FIRST_PRINT=False##
        # cnt = 0
        sample_id = sample['id']
        system = sample['system']
        hallucination_type = sample['hallucination_type']
        # Prepare inputs and labels
        max_length = 256  # Define a consistent max_length for both input and labels

        input_ids = self.tokenizer(sample["docs"], truncation=True, padding='longest', return_tensors="pt").input_ids.to(self.device)
        labels = self.tokenizer(sample["machine_summary"], truncation=True, padding='longest', return_tensors="pt").input_ids.to(self.device)

        # Generate outputs using the model
        outputs = self.model.generate(input_ids=input_ids, max_length=50, do_sample=True, top_p=0.9).detach().cpu().numpy()

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Print sample input, prediction, and gold summary
        # print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")
        # print(f"Generated summary:\n{decoded_preds[0]}")
        # print(f"Reference summary:\n{decoded_labels[0]}")

        # Metrics calculation

        ################
        ## BERT SCORE ##
        ################
        # print('---'* 20)
        bertscore_result = self.bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        del bertscore_result['hashcode']
        tmp={}
        for k, v_list in bertscore_result.items():
            tmp["BertScore "+k] = np.mean(v_list)
        bertscore_result = tmp

        ################
        ##   rouge    ##
        ################
        # print('---'* 20)
        rouge_result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)

        ################
        ##    MQAG    ##
        ################
        # print('---'* 20)
        # print(np.shape(decoded_preds[0]))
        # print(decoded_preds)
        # print(decoded_preds[0])
        # print('-'*10)
        # print(decoded_labels)
        # print(decoded_labels[0])
        try:
            mqag_score = self.mqag_model.score(candidate=decoded_preds[0], reference=decoded_labels[0], num_questions=3, verbose=False)
        except:
            print('retry')
            try:
                mqag_score = self.mqag_model.score(candidate=decoded_preds[0], reference=decoded_labels[0], num_questions=3, verbose=False)
            except:
                print('error while running mqag')
                return eval_results_copy
                
        # print(score['total_variation'])
        tmp={}
        for k, v in mqag_score.items():
            tmp["MQAG "+k] = v
        mqag_score = tmp

        ###################
        ## SelfCheck-NLI ##
        ###################
        # print('---'* 20)
        # Generate outputs using the self.model
        # sample
        sample1 = self.model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()
        sample2 = self.model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()
        sample3 = self.model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_p=0.9).detach().cpu().numpy()

        # Decode predictions and labels
        sample1 = self.tokenizer.batch_decode(sample1, skip_special_tokens=True)[0]
        sample2 = self.tokenizer.batch_decode(sample2, skip_special_tokens=True)[0]
        sample3 = self.tokenizer.batch_decode(sample3, skip_special_tokens=True)[0]

        sent_scores_nli = self.selfcheck_nli.predict(
            sentences = decoded_preds[0],                          # list of sentences
            sampled_passages = [sample1, sample2, sample3], # list of sampled passages
        )
        # print(sent_scores_nli)
        num_nli_contr = 0
        nli_threshold = 0.5397 ## https://github.com/potsawee/selfcheckgpt/issues/17
        for n in sent_scores_nli:
            if n < nli_threshold:
                num_nli_contr+=1
        nli_score = {"NLI Score": np.mean(sent_scores_nli), "NLI contradiction %": float(num_nli_contr/len(sent_scores_nli))}

        # print('\n'+'---'* 20)

        # Aggregate results
        for k, v in rouge_result.items():
            eval_results[(sample_id, system)]["Rouge " + k].append(v)
        for k, v in bertscore_result.items():
            eval_results[(sample_id, system)][k].append(v)
        for k, v in mqag_score.items():
            eval_results[(sample_id, system)][k].append(v)
        eval_results[(sample_id, system)]["NLI Score"].append(nli_score["NLI Score"])
        eval_results[(sample_id, system)]["NLI contradiction %"].append(nli_score["NLI contradiction %"])
        eval_results[(sample_id, system)]["hallucination_type"].append(hallucination_type[0])

        return eval_results

    def write_results_to_csv(self, eval_results):
        """Write the evaluation results to a CSV file."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'{self.storing_name}_{current_time}.csv'
        with open(file_name, 'w', newline='') as csvfile:
            # Dynamically extract fieldnames from the collected eval_results
            fieldnames = ['sample_id', 'system'] + list(next(iter(eval_results.values())).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for (sample_id, system), metrics in eval_results.items():
                row = {'sample_id': sample_id}
                row['system'] = system 
                for metric, values in metrics.items():
                    try:
                        row[metric] = np.mean(values)  # Or use another appropriate method of aggregation
                    except:
                        row[metric] = values
                writer.writerow(row)

xsum_hal = XSum_Hallucination()
# xsum_hal.print_statistic()
# xsum_hal.save_statistics_to_excel()  # This will save the statistics to an Excel file.
# xsum_hal.nli_evaluation()
xsum_hal.import_and_evaluate('./xsum_none_hal.npy')