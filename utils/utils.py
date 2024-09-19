# Importing libraries
from openai import AzureOpenAI
import json
import hashlib

def get_embeddings_vector(client: AzureOpenAI, string: str) -> list:
    """
    Retrieves the embeddings vector for a given string using OpenAI.
    
    Args:
        client (AzureOpenAI): Azure OpenAI client.
        string (str): The input string to get embeddings for.
        
    Returns:
        list: The resulting embeddings vector.
    """
    # Checking input type
    if not isinstance(string, str):
        raise ValueError("The argument 'string' must be a string.")
    
    # Request the embeddings from the OpenAI model
    response = client.embeddings.create(
        input=[string],
        model='text-embedding-ada-002',
    )
    
    # Parse the response and return the embedding
    response_json = json.loads(response.model_dump_json(indent=2))
    return response_json['data'][0]['embedding']


def get_vector_id(sentence: str, metadata: dict) -> str:
    """
    Generates a unique vector ID using a hash based on the sentence and its metadata.
    
    Args:
        sentence (str): The sentence or text to use for the vector ID.
        metadata (dict): Metadata dictionary containing additional information (e.g., title, authors).
        
    Returns:
        str: The generated vector ID as a hash string.
    """
    if not isinstance(sentence, str):
        raise ValueError("The argument 'sentence' must be a string.")
    if not isinstance(metadata, dict):
        raise ValueError("The argument 'metadata' must be a dictionary.")
    
   