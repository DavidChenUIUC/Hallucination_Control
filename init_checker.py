import torch, spacy, evaluate
import numpy as np

from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore
from selfcheckgpt.modeling_mqag import MQAG

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # Import for LLM model
from peft import PeftModel, PeftConfig  # Import for Peft

torch.manual_seed(28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_mqag = SelfCheckMQAG(device=device)
selfcheck_bertscore = SelfCheckBERTScore()
print(f"|- Using {device} device")

# Load peft model
peft_model_id = "./lora-flan-t5-xxl-1e-3_seed1001/checkpoint-500"
config = PeftConfig.from_pretrained(peft_model_id)

# Load base LLM model and tokenizer
base_model_name_or_path = config.base_model_name_or_path  # Replace with your base model path if necessary
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path, load_in_8bit=True, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

# Load the Peft model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()
model.to(device)
print("Peft model loaded")

# Load MQAG model
mqag_model = MQAG(
    g1_model_type='race', # race (more abstractive), squad (more extractive)
    device=device
)

# Load Rouge metric
metric = evaluate.load("rouge")


passage = '''Beatrice: I am in town, shopping. They have nice scarfs in the shop next to the church. Do you want one? Leo: No, thanks Beatrice: But you don't have a scarf. Leo: Because I don't need it. Beatrice: Last winter you had a cold all the time. A scarf could help. Leo: I don't like them. Beatrice: Actually, I don't care. You will get a scarf. Leo: How understanding of you! Beatrice: You were complaining the whole winter that you're going to die. I've had enough. Leo: Eh.'''  # Replace with the actual passage

gold = '''Beatrice wants to buy Leo a scarf, but he doesn't like scarves. She cares about his health and will buy him a scarf no matter his opinion.'''

input_ids = tokenizer(passage, return_tensors="pt", truncation=True).input_ids.to(device)
gold_ids = tokenizer(gold, return_tensors="pt", truncation=True).input_ids.to(device)

def sampling(num_samples=3):
    samples=[]

    # Generating predictions and appending to samples
    for i in range(num_samples):
        outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{i}th prediction: {pred_text}")
        samples.append(pred_text)
    return samples


def mqag_consistancy_check():
    global gold_ids, gold
    print("|- Running mqag_consistancy_check")
    samples = sampling()
    best_var = float('inf')
    best_sample = ''
    gold_text = tokenizer.decode(gold_ids[0], skip_special_tokens=True)

    samples = sampling()

    best_var = float('inf')
    best_sample = ''
    for i, pred_text in enumerate(samples):
        # Compute MQAG score
        score = mqag_model.score(candidate=pred_text, reference=passage, num_questions=len(samples), verbose=True)
        
        # Compute Rouge score
        rouge_result = metric.compute(predictions=[pred_text], references=[gold_text], use_stemmer=True)

        # Extract and print scores for analysis
        kl_div = score['kl_div']
        counting = score['counting']
        hellinger = score['hellinger']
        total_variation = score['total_variation']

        rouge1 = rouge_result['rouge1']* 100
        rouge2 = rouge_result['rouge2']* 100
        rougeL = rouge_result['rougeL'] * 100
        rougeLsum = rouge_result['rougeLsum'] * 100

        print(f"-----{i}th sample-----")
        print("KL-div    =", kl_div)
        print("Counting  =", counting)
        print("Hellinger =", hellinger)
        print("Total Var =", total_variation)
        print(f"Rouge1: {rouge1:.2f}%")
        print(f"Rouge2: {rouge2:.2f}%")
        print(f"RougeL: {rougeL:.2f}%")
        print(f"RougeLsum: {rougeLsum:.2f}%")
    

def get_bertscores_for_multi_samples():

    nlp = spacy.load("en_core_web_sm")
    sentences = [sent for sent in nlp(passage).sents]  # List[spacy.tokens.span.Span]
    sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
    print("SelfCheck running on {} sentences...".format(len(sentences)))

    sent_scores_mqag = selfcheck_mqag.predict(
        sentences,
        passage,
        samples,
        num_questions_per_sent=5,
        scoring_method='bayes_with_alpha',
        beta1=0.95, beta2=0.95,
    )

    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences,
        samples,
    )

    # Unedited passage
    print("MQAG\tBERTScore")
    for s1, s2 in zip(sent_scores_mqag, sent_scores_bertscore):
        print("{:.4f}\t{:.4f}".format(s1, s2))
        
if __name__=='__main__':
    mqag_consistancy_check()