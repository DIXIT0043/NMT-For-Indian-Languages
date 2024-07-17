import torch
from typing import Union
from torch.utils.data import DataLoader
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import evaluate

# Paths to the tokenizer and model
tokenizer_path = r"C:\Users\kumar\OneDrive\Desktop\Final\Neural-Machine-Translation-and-Large-Language-Models-to-Bridge-Indian-Vernaculars\tokenizer_hi-te"
model_path = r"C:\Users\kumar\OneDrive\Desktop\Final\Neural-Machine-Translation-and-Large-Language-Models-to-Bridge-Indian-Vernaculars\translation-hi-te"

# Load evaluation metric
bleu = evaluate.load("sacrebleu")

# Inference function
def inference(text: Union[str, list]) -> list:
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_path)
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    
    tokenizer.src_lang = "hi_IN"
    tokenizer.tgt_lang = "te_IN"
    
    if isinstance(text, str):
        text = [text]
    
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
    translation = model.generate(**tokenized_text, forced_bos_token_id=tokenizer.lang_code_to_id["te_IN"])
    
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    return translated_text

# Batched inference function
def batched_inference(texts: list, batch_size: int = 6) -> list:
    pred_loader = DataLoader(texts, batch_size=batch_size)
    translated_text = []
    
    for batch in pred_loader:
        temp_translate = inference(text=batch)
        translated_text.extend(temp_translate)
        
    return translated_text

# Calculate BLEU score
def calculate_bleu(predictions: list, references: list) -> float:
    results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    return results['score']

# Main function
if __name__ == '__main__':
    # Replace these example texts and references with your actual data
    texts = ["आप कैसे हैं"]
    references = ["ఎలా మీరు"]

    # Perform batched inference
    translations = batched_inference(texts=texts)
    print(f"Hindi text is :{texts}Translations is :{translations}")
'''    # Calculate BLEU score
    bleu_score = calculate_bleu(translations, references)
    
    print(f"Hindi text is :{texts}Translations is :{translations}")
    print("BLEU Score:", bleu_score)'''
    

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# Calculate BLEU score
smoothie = SmoothingFunction().method4
bleu_score = sentence_bleu(references=references,hypothesis=translations, smoothing_function=smoothie)

print(f"BLEU score: {bleu_score}")