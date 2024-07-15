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
    texts = [
        "मुझे भूख लगी है",  # "I'm hungry"
        "मेरा नाम अजय है",  # "My name is Ajay"
        "यह एक सुंदर दिन है",  # "It is a beautiful day"
        "क्या तुम मुझे सुन सकते हो?",  # "Can you hear me?"
        "मुझे हिंदी सीखने में मजा आता है"  # "I enjoy learning Hindi"
    ]
    references = [
        "నాకు ఆకలిగా ఉంది",  # "I'm hungry"
        "నా పేరు అజయ్",  # "My name is Ajay"
        "ఇది ఒక అందమైన రోజు",  # "It is a beautiful day"
        "మీకు నాకు వినిపిస్తుందా?",  # "Can you hear me?"
        "నాకు హిందీ నేర్చుకోవడం ఇష్టం"  # "I enjoy learning Hindi"
    ]


    # Perform batched inference
    translations = batched_inference(texts=texts)
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(translations, references)
    
    print("Translations:", translations)
    print("BLEU Score:", bleu_score)