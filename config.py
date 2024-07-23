#from datasets import load_metric
import evaluate
metric = evaluate.load("sacrebleu")
# Key Parameter
from_and_to_nmt = "hindi-to-bengali"
batch_size = 16
MAX_LEN = 40
# LLM checkpoints settings
model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
DATAPATH = r"C:\Users\kumar\OneDrive\Desktop\Final\Neural-Machine-Translation-and-Large-Language-Models-to-Bridge-Indian-Vernaculars\NMT_data_cleaned.csv"

if from_and_to_nmt == "hindi-to-bengali":
    INPUT_LANG = "hindi"
    TARGET_LANG = "bengali"
    src_lang = 'hi_IN'
    trg_lang = 'bn_IN'
    df_src = 'hindi'
    df_trg = 'bengali'
    save_model = 'translation-hi-bn'
    token = 'tokenizer_hi-bn'

# Language settings based on translation direction
elif from_and_to_nmt == "hindi-to-tamil":
    INPUT_LANG = "hindi"
    TARGET_LANG = "tamil"
    src_lang = 'hi_IN'
    trg_lang = 'ta_IN'
    df_src = 'hindi'
    df_trg = 'tamil'
    save_model = 'translation-hi-gu'
    token = 'tokenizer_hi-ta'
    
# Language settings based on translation direction
elif from_and_to_nmt == "hindi-to-gujrati":
    INPUT_LANG = "hindi"
    TARGET_LANG = "gujrati"
    src_lang = 'hi_IN'
    trg_lang = 'gu_IN'
    df_src = 'hindi'
    df_trg = 'gujrati'
    save_model = 'translation-hi-gu'
    token = 'tokenizer_hi-gu'
    
elif from_and_to_nmt == 'hindi-to-telugu':
    INPUT_LANG = "hindi"
    TARGET_LANG = "telugu"
    src_lang = 'hi_IN'
    trg_lang = 'te_IN'
    df_src = 'hindi'
    df_trg = 'telugu'
    save_model = 'translation-hi-te'
    token = 'tokenizer_hi-te'

elif from_and_to_nmt == "hindi-to-punjabi":
    INPUT_LANG = "hindi"
    TARGET_LANG = "punjabi"
    src_lang = 'hi_IN'
    trg_lang = 'pa_IN'
    df_src = 'hindi'
    df_trg = 'punjabi'
    save_model = 'translation-hi-pa'

elif from_and_to_nmt == "hindi-to-sanskrit":
    INPUT_LANG = "hindi"
    TARGET_LANG = "sanskrit"
    src_lang = 'hi_IN'
    trg_lang = 'sa_IN'
    df_src = 'hindi'
    df_trg = 'sanskrit'
    save_model = 'translation-hi-sa'
    token = 'tokenizers-hi-sa'

else:
    print("Please modify your config file.")