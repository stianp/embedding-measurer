import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Paths
data_dir = os.path.join(os.path.dirname(__file__), '../data')
text_file_path = os.path.join(data_dir, 'ai-texts.json')
embeddings_file_path = os.path.join(data_dir, 'embeddings.json')

def load_texts(file_path):
    """
    Load texts and their IDs from the specified JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing texts.
        
    Returns:
        list: A list of dictionaries with 'id' and 'text'.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        return [{'id': entry['id'], 'text': entry['text']} for entry in data]

def generate_embedding(text, model="text-embedding-3-small"):
    """
    Generate an embedding for a given text using OpenAI's API.
    
    Args:
        text (str): The text to generate an embedding for.
        model (str): The model to use for embedding generation.
        
    Returns:
        list: The generated embedding vector.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        return response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None

def generate_embeddings(texts_with_ids, model="text-embedding-3-small"):
    """
    Generate embeddings for a list of texts with their IDs.
    
    Args:
        texts_with_ids (list): A list of dictionaries with 'id' and 'text'.
        model (str): The model to use for embedding generation.
        
    Returns:
        list: A list of dictionaries with 'id' and 'embedding'.
    """
    embeddings = []
    for i, item in enumerate(texts_with_ids):
        embedding = generate_embedding(item['text'], model=model)
        if embedding is not None:
            embeddings.append({'id': item['id'], 'embedding': embedding})
            print(f"Generated embedding for text {i + 1}/{len(texts_with_ids)}")
    return embeddings

def save_embeddings(file_path, embeddings):
    """
    Save embeddings to a JSON file.
    
    Args:
        file_path (str): Path to the JSON file to save embeddings.
        embeddings (list): A list of embeddings to save.
    """
    with open(file_path, 'w') as file:
        json.dump(embeddings, file, indent=2)

def main():
    # Load texts from the JSON file
    texts_with_ids = load_texts(text_file_path)
    
    # Generate embeddings for the texts
    embeddings = generate_embeddings(texts_with_ids)
    
    # Save the embeddings to a JSON file
    save_embeddings(embeddings_file_path, embeddings)

if __name__ == "__main__":
    main()
