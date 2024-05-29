import os
import json
import re
from bs4 import BeautifulSoup

def read_data_from_file(file_path):
    """
    Read text data from a local file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Content of the file as a single string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_data_from_json(json_path):
    """
    Read text data from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: List of texts.
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_text(html_content):
    """
    Clean and preprocess HTML content.

    Args:
        html_content (str): Raw HTML content.

    Returns:
        str: Cleaned text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

def preprocess_texts(texts):
    """
    Preprocess a list of texts.

    Args:
        texts (list of str): List of raw texts.

    Returns:
        list of str: List of cleaned and preprocessed texts.
    """
    return [clean_text(text) for text in texts]

def save_preprocessed_data(data, file_path):
    """
    Save preprocessed data to a file.

    Args:
        data (list of str): List of preprocessed texts.
        file_path (str): Path to save the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file)

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        list of str: List of preprocessed texts.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    # Example usage
    # Read raw data
    raw_texts = read_data_from_json('data/hardcoded_texts.json')
    
    # Preprocess the data
    preprocessed_texts = preprocess_texts(raw_texts)
    
    # Save preprocessed data
    save_preprocessed_data(preprocessed_texts, 'data/preprocessed_texts.json')
    
    # Load preprocessed data
    loaded_texts = load_preprocessed_data('data/preprocessed_texts.json')
    print(loaded_texts)
