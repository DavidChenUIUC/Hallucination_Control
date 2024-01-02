import json, time, openai, torch
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
import pandas as pd
from collections import defaultdict
from selfcheckgpt.modeling_mqag import MQAG
import random
warnings.filterwarnings("ignore")


class QA(BaseModel):
    answer: Literal[Literal['A', 'B', 'C', 'D']]

seed = 1234                    
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

class QA_GPT:
    def __init__(self, num_questions=3):
        # self.GPT_MODEL = "TURN OFF"
        # self.GPT_MODEL = "gpt-4-1106-preview"
        self.GPT_MODEL = "gpt-3.5-turbo-16k"
        self.tools = self.get_tool_list()
        self.system_result = defaultdict(list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mqag_model = MQAG(
            g1_model_type='race', # race (more abstractive), squad (more extractive)
            device=device
        )
        self.num_questions = num_questions
        self.option_mapping = {'A': 0 , 'B': 1, 'C': 2, 'D': 3}

    def get_tool_list(self):
        schema = QA.schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "QA",
                    "description": "Answer the question with A, B, C, D based on the context and question.",
                    "parameters": schema
                }
            }
        ]
        return tools

    def parse_resp(self, document, summary, question, options, seed=None):
        # print(f"Document: {document}")
        # print(f"Summary: {summary}")
        messages = [
            {
                "role": "system",
                "content": f"Your task is to answer the question based on the context. Do no make assumptions. Only response in Literal['A', 'B', 'C', 'D']."

            },{
                "role": "user",
                "content": f"Context: {summary}, Questions: {question}, Options {options}. Let's think it in a step-by-step way but still only response in Literal['A', 'B', 'C', 'D']."
                # Poor: "content": f"Context: {document}, Summary: {summary}. Is the summary supported by the context? Let's work this out in a step by step way to be sure we have the right answer."
            }
        ]
        chat_response = self.chat_completion_request(messages, tools=self.tools, seed=seed)
        try:
            assistant_message = chat_response.json()["choices"][0]["message"]
            # pprint(assistant_message)
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
            json_data.update({"seed": seed})
            
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
       
    def qa(self, document, summary):
        start = time.time()
        question_items = self.mqag_model.generate(context=summary, do_sample=True,num_questions=self.num_questions)
        questions = []
        answers = []
        options = [] # 2D list
        for question_item in question_items:
            question = question_item['question']
            option = [question_item['options'][i] for i in range(4)]

            #  Ask GPT to answer the question
            try:
                arguments_dict = self.parse_resp(document, summary, question, option)
                # arguments_dict = {'hallucination_type': 'ERROR', 'hallucination_level': -1, 'confidence_score': -1}
            except:
                try:
                    arguments_dict = self.parse_resp(document, summary, question, option, seed=random.randint(0, 100000))
                except:
                    print("** ERROR ** error while communicating with GPT-3 API")
                    continue
            
            # Get the answer
            try:
                answers.append(self.option_mapping[arguments_dict['answer']])
                questions.append(question)
                options.append(option)

            except:
                print("** ERROR ** error while parsing GPT-3 response")
        end = time.time()
        execution_time = round((end - start),2)
        return questions, options, answers, execution_time