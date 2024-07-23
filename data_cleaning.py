import re
import pandas as pd

def keep_hindi_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0900-\u097F\s\.,!?;:\'\"-]', '', text)
    return ''

def keep_bengali_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0980-\u09FF\s\.,!?;:\'\"-]', '', text)
    return ''

def keep_telugu_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0C00-\u0C7F\s\.,!?;:\'\"-]', '', text)
    return ''

def keep_tamil_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0B80-\u0BFF\s\.,!?;:\'\"-]', '', text)
    return ''

def keep_gujarati_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\u0A80-\u0AFF\s\.,!?;:\'\"-]', '', text)
    return ''

def clean_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop rows with any missing values
    data.dropna(how="any", inplace=True)

    # Apply the cleaning functions to each language column
    data['hindi'] = data['hindi'].apply(keep_hindi_text)
    data['bengali'] = data['bengali'].apply(keep_bengali_text)
    data['telugu'] = data['telugu'].apply(keep_telugu_text)
    data['tamil'] = data['tamil'].apply(keep_tamil_text)
    data['gujrati'] = data['gujrati'].apply(keep_gujarati_text)

    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    return data

# Usage
file_path = r'C:\Users\kumar\OneDrive\Desktop\Final\Neural-Machine-Translation-and-Large-Language-Models-to-Bridge-Indian-Vernaculars\NMT_data_8.csv'
cleaned_data = clean_dataset(file_path)
cleaned_data.to_csv('NMT_data_cleaned.csv', index=False)