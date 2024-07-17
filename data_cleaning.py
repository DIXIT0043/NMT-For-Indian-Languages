import re
import pandas as pd

def keep_hindi_text(text):
    hindi_text = re.sub(r'[^\u0900-\u097F\s\.,!?;:\'\"-]', '', text)
    return hindi_text

def keep_bengali_text(text):
    bengali_text = re.sub(r'[^\u0980-\u09FF\s\.,!?;:\'\"-]', '', text)
    return bengali_text

def keep_telugu_text(text):
    telugu_text = re.sub(r'[^\u0C00-\u0C7F\s\.,!?;:\'\"-]', '', text)
    return telugu_text

def keep_tamil_text(text):
    tamil_text = re.sub(r'[^\u0B80-\u0BFF\s\.,!?;:\'\"-]', '', text)
    return tamil_text

def keep_gujarati_text(text):
    gujarati_text = re.sub(r'[^\u0A80-\u0AFF\s\.,!?;:\'\"-]', '', text)
    return gujarati_text

def keep_punjabi_text(text):
    punjabi_text = re.sub(r'[^\u0A00-\u0A7F\s\.,!?;:\'\"-]', '', text)
    return punjabi_text

def clean_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Apply the cleaning functions to each language column
    data['hindi'] = data['hindi'].apply(keep_hindi_text)
    data['bengali'] = data['bengali'].apply(keep_bengali_text)
    data['telugu'] = data['telugu'].apply(keep_telugu_text)
    data['tamil'] = data['tamil'].apply(keep_tamil_text)
    data['gujrati'] = data['gujrati'].apply(keep_gujarati_text)
    data['punjabi'] = data['punjabi'].apply(keep_punjabi_text)

    # Remove rows with any missing values
    data.dropna(how="any", inplace=True)

    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    return data

# Usage
file_path = r'C:\Users\kumar\OneDrive\Desktop\Final\Neural-Machine-Translation-and-Large-Language-Models-to-Bridge-Indian-Vernaculars\NMT_data.csv'
cleaned_data = clean_dataset(file_path)
cleaned_data.to_csv('NMT_data_cleaned.csv', index=False)
