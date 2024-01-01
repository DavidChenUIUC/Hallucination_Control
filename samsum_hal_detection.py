import json, time, openai
import requests
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, confloat
from typing import Literal
import warnings
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from datasets import load_dataset

import random
import pandas as pd  # Ensure 'pandas' and 'openpyxl' are installed.
from collections import defaultdict
warnings.filterwarnings("ignore")


class HallucinationDetection(BaseModel):
    hallucination_type: Literal['intrinsic', 'extrinsic', 'both', 'NULL']
    hallucination_level: confloat(ge=0, le=1)
    confidence_score: confloat(ge=0, le=1)

class SamSum_Hallucination_Detection:
    def __init__(self):
        # self.GPT_MODEL = "TURN OFF"
        # self.GPT_MODEL = "gpt-4-1106-preview"
        self.GPT_MODEL = "gpt-3.5-turbo-16k"
        self.tools = self.get_tool_list()
        self.result = defaultdict(lambda: defaultdict(str) )
        self.test_hal = True
        
        self.split = 'test'
        dataset = load_dataset("samsum")[self.split]
        self.dataset = defaultdict(lambda: defaultdict(str) )
        for sample in dataset:
            self.dataset[sample['id']] = sample
            

        # set seed
        seed = 2024
        np.random.seed(seed)
        random.seed(seed)
        
        self.reducted_csv = 'sumsum_hal_reduction_results'

    def get_tool_list(self):
        schema = HallucinationDetection.schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "HallucinationDetection",
                    "description": "Detect the hallucination type, level and confidence of determining the hallucination in the summary of a document. ",
                    "parameters": schema
                }
            }
        ]
        return tools

    def parse_resp(self, document, summary, seed = None):
        # print(f"Document: {document}")
        # print(f"Summary: {summary}")
        messages = [
            {
                "role": "system",
                "content": f"Your task is to determine if there is any hallucinations in the given summarization with the given context.  Do no make assumptions. Use function calling and only response in JSON with hallucination_type: Literal['intrinsic', 'extrinsic', 'both', 'NULL'] hallucination_level: confloat(ge=0, le=1), confidence_score: confloat(ge=0, le=1)"

            },{
                "role": "user",
                "content": f"Context: {document}, Summary: {summary}. Is the summary supported by the context?"
                # Poor: "content": f"Context: {document}, Summary: {summary}. Is the summary supported by the context? Let's work this out in a step by step way to be sure we have the right answer."
            }
        ]
        chat_response = self.chat_completion_request(messages, tools=self.tools, seed=seed)
        try:            
            pprint(chat_response.json())
            assistant_message = chat_response.json()["choices"][0]["message"]
        except Exception as e:
            print("Unable to parse ChatCompletion response")
            print(f"Error: {chat_response.json()}")
            return e
        arguments_dict = json.loads(assistant_message['tool_calls'][0]['function']['arguments'])
        return arguments_dict

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages, tools=None, tool_choice='auto', seed=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + "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",
        }
        json_data = {"model": self.GPT_MODEL, "messages": messages, "seed": 123, "temperature": 0.2 }#,  "response_format": {"type": "json_object"}}
        if tools is not None:
            json_data.update({"tools": tools, "tool_choice":tool_choice})
        if seed is not None:
            json_data.update({"seed":seed})
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    def ask_gpt(self):
        # Load data from Excel
        df = pd.read_csv(self.reducted_csv, keep_default_na=False)
        

        for _, row in tqdm(df.iterrows(), desc = f"Evaluating", total=len(df)):
            document_id = row['id']
            # document = row['dialogue']
            summary = row['summary']
            original_dialogue = self.dataset[str(document_id)]['dialogue']
            original_summary = self.dataset[str(document_id)]['summary']
            
            print(original_dialogue)
            print(original_summary)

            start = time.time()
            retry_limit =3
            
            try:
                arguments_dict = self.parse_resp(original_dialogue, summary)
                ERROR = False
            except:
                ERROR = True
                
            while ERROR and retry_limit >0:
                try:
                    arguments_dict = self.parse_resp(original_dialogue, summary, seed = random.randint(0, 1000000))
                    print(arguments_dict)
                    ERROR = False
                except:
                    print("|- retrying")
                    arguments_dict = {'hallucination_type': 'ERROR', 'hallucination_level': -1, 'confidence_score': -1}
                    retry_limit-=1
            if ERROR:
                print("** ERROR ** error while communicating with GPT-3 API")

            retry_limit=3
            try:
                org_arguments_dict = self.parse_resp(original_dialogue, original_summary)
                ERROR = False
            except:
                ERROR = True
                
            while ERROR and retry_limit >0:
                try:
                    org_arguments_dict = self.parse_resp(original_dialogue, original_summary, seed = random.randint(0, 1000000))
                    ERROR = False
                except:
                    print("|- retrying")
                    org_arguments_dict = {'hallucination_type': 'ERROR', 'hallucination_level': -1, 'confidence_score': -1}
                    retry_limit-=1
            if ERROR:
                print("** ERROR ** error at original_summary while communicating with GPT-3 API")
            end = time.time()

            time_taken = end - start
            
            result = {
                'id': document_id,
                'dialogue': original_dialogue, 
                'summary': summary,
                'hallucination_type': arguments_dict['hallucination_type'],
                'hallucination_level': arguments_dict['hallucination_level'],
                'confidence_score': arguments_dict['confidence_score'],

                'org_summary': original_summary,
                'org_hallucination_type': org_arguments_dict['hallucination_type'],
                'org_hallucination_level': org_arguments_dict['hallucination_level'],
                'org_confidence_score': org_arguments_dict['confidence_score'],

                'prediction': True if arguments_dict['hallucination_type'] != 'NULL' else False,
                'org_prediction': True if org_arguments_dict['hallucination_type'] != 'NULL' else False,

                'time_taken': round(time_taken, 4),
                'gpt_model': self.GPT_MODEL
            }
                
            # Append the result for this document to the list for the system.
            self.result['id'] = result

            break
            # cnt+=1
            # if cnt == 30:
            #     self.save_statistics_to_excel()

        # Save the result to excel
        self.save_statistics_to_excel()
        
    def save_statistics_to_excel(self, file_name='samsum_red_gpt3_predict_results.csv'):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = file_name.replace('.csv', f'_{current_time}.csv')
        
        # Create a list of dictionaries. Each dictionary represents a row in the CSV.
        rows = []
        for result in self.result.values():
            print(result)
            rows.append({
                'id': result['id'],
                'dialogue': result['dialogue'],
                'predict_summary': result['summary'],
                'hallucination_type': result['hallucination_type'],
                'hallucination_level': result['hallucination_level'],
                'confidence_score': result['confidence_score'],

                'org_summary': result['org_summary'],
                'org_hallucination_type': result['org_hallucination_type'],
                'org_hallucination_level': result['org_hallucination_level'],
                'org_confidence_score': result['org_confidence_score'],
                
                'prediction': result['prediction'],
                'org_prediction': result['org_prediction'],
                
                'time_taken': result['time_taken'],
                'gpt_model': result['gpt_model']
            })
        
        # Convert the list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(rows)
        
        # Save the DataFrame to a CSV file
        df.to_csv(file_name, index=False)
        
        print(f"Results saved to {file_name}")

detector = SamSum_Hallucination_Detection()
detector.ask_gpt()
