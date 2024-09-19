import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

class EmbeddingDB:
    """
    Class to create and manage an embedding database from sentences extracted from scientific papers.
    
    Attributes:
        index_name (str): The name of the index.
        data (list): List of sentences and their metadata.
        embeddings (AzureOpenAIEmbeddings): Azure embeddings client.
        client (AzureOpenAI): Azure OpenAI client.
    """
    def __init__(self, index_name: str):
        """
        Initializes the EmbeddingDB with the index name.
        
        Args:
            index_name (str): The name of the index.
        """
        self.index_name = index_name
        self.data = []  # List to store sentence and metadata information
        self.embeddings = AzureOpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            openai_api_type="azure",
            api_key=os.getenv("OPENAI_KEY")
        )
        self.client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_KEY")
        )

    def save_sentences_into_database(self):
        """
        Saves the sentences and their embeddings into the database.
        
        This method assumes that `self.data` contains a list of dictionaries, where each dictionary
        represents a sentence and its metadata. Each sentence is used to create an embedding, and
        both the embedding and the metadata are stored in the FAISS index.
        """
        for entry in self.data:
            sentence = entry["sentence"]
            metadata = entry["metadata"]

            # Create embedding for the sentence
            embeds = utils.get_embeddings_vector(self.client, sentence)

            # Add the sentence itself to the metadata for later retrieval
            metadata["sentence"] = sentence

            # Save the embedding and metadata into the database
            self.save_vector_and_meta(sentence, embeds, metadata)

    def save_vector_and_meta(self, sentence: str, embeds: list, metadata: dict):
        """
        Saves the embedding vectors and their metadata into the index.
        
        Args:
            sentence (str): The sentence to save.
            embeds (list): The embedding vectors.
            metadata (dict): The metadata for the sentence.
            
        Returns:
            list: List of IDs added to the index.
        """
        try:
            # Store the sentence and its embedding
            text_embeddings = [(sentence, embeds)]
            metadata_list = [metadata]
            
            # Create an ID for the vector (you can customize this logic if needed)
            vector_id = utils.get_vector_id(sentence, metadata)
            ids = [vector_id]

            if hasattr(self, 'index'):
                ids_added = self.index.add_embeddings(text_embeddings, metadata_list, ids)
            else:
                # Create a new FAISS index if it does not exist
                self.index = FAISS.from_documents([Document(page_content=sentence, metadata=metadata)], self.embeddings)
                ids_added = [vector_id]

            return ids_added

        except Exception as e:
            print(f"Exception of type {type(e).__name__}: {e}")
            if type(e).__name__ == "ValueError" and "Tried to add ids that already exist" in str(e):
                return
            else:
                raise e
            
    def save_index_locally(self) -> bool:
        """
        Saves the current FAISS index to a local file.
        
        Returns:
            bool: True if the index was saved successfully, False otherwise.
        """
        try:
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'databases', self.index_name)
            if self.index:
                self.index.save_local(save_path)
            return True
        except Exception as e:
            print(f"Exception occurred while saving the index locally: {e}")
            return False

    def load_existing_index(self, allow_dangerous_deserialization: bool = False) -> bool:
        """
        Loads an existing FAISS index from a local file.
        
        Args:
            allow_dangerous_deserialization (bool): Whether to allow unsafe deserialization.
        
        Returns:
            bool: True if the index was loaded successfully, False otherwise.
        """
        try:
            load_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'databases', self.index_name)
            self.index = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)
            return True
        except Exception as e:
            print(f"Exception occurred while loading the index: {e}")
            return False