import openai
import json
import uuid
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure the data directory exists
data_dir = os.path.join(os.path.dirname(__file__), '../data')
os.makedirs(data_dir, exist_ok=True)
data_file_path = os.path.join(data_dir, 'ai-texts.json')

def generate_texts_for_topic(topic, num_texts=5):
    """
    Generate a specified number of texts for a given topic using OpenAI's GPT-3.5-turbo.
    
    Args:
        topic (str): The topic for text generation.
        num_texts (int): The number of texts to generate.
        
    Returns:
        list: A list of generated texts with topic labels and unique identifiers.
    """
    texts = []
    for i in range(num_texts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a brilliant, creative writer who approaches topics from unconventional, even orthogonal angles. You studied Austrian economics and complexity economics, and often use concepts from those subjects in your article. You write in a conversational, and approachable way."},
                    {"role": "user", "content": f"Write an article about {topic}."}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            text_content = response['choices'][0]['message']['content']
            text_entry = {
                "id": str(uuid.uuid4()),
                "topic": topic,
                "text": text_content
            }
            texts.append(text_entry)
            
            # Write the text entry to the JSON file immediately
            append_to_json(data_file_path, text_entry)
            print(f"Text number {i + 1} for topic '{topic}' generated and stored.")
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            continue
    return texts

def append_to_json(file_path, data):
    """
    Append a single data entry to a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        data (dict): Data to append.
    """
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=2)
    else:
        with open(file_path, 'w') as file:
            json.dump([data], file, indent=2)

def generate_and_store_texts():
    """
    Generate texts for predefined topics and store them in a JSON file.
    """
    topics = ["gambling on casinos", "wedding planning", "having a baby", "parenthood"]
    for topic in topics:
        generate_texts_for_topic(topic, num_texts=5)

if __name__ == "__main__":
    generate_and_store_texts()
