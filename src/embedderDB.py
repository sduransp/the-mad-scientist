# Importing libraries
import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from typing import List, Dict
import ast

class EmbeddingDBFromData:
    """
    Class to create and manage an embedding database from sentences and their metadata.

    Attributes:
        index_name (str): The name of the index.
        embeddings (OpenAIEmbeddings): Azure OpenAI embeddings client.
        index (FAISS): The FAISS vector store index.
        data_list (List[Dict]): List of dictionaries containing sentences and metadata.
    """

    def __init__(self, index_name: str, data_variable: str):
        """
        Initializes the EmbeddingDBFromData with the index name and data variable.

        Args:
            index_name (str): The name of the index.
            data_variable (str): The string representation of the data containing sentences and metadata.
        """
        self.index_name = index_name

        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_version="2024-02-01",
            azure_endpoint="https://genai-nexus.api.corpinter.net/apikey/",
            openai_api_type="azure",
            api_key=os.getenv("OPENAI_ADA")
        )
        self.client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint="https://genai-nexus.api.corpinter.net/apikey/",
            api_key=os.getenv("OPENAI_ADA")
        )

        self.data_variable = data_variable
        self.index = None  # Initialize the index attribute

    def parse_data_variable(self) -> List[Dict]:
        """
        Parses the data_variable string into a list of dictionaries.

        Returns:
            List[Dict]: The parsed data list.
        """
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        data_list = ast.literal_eval(self.data_variable)
        return data_list

    def create_embeddings_and_index(self):
        """
        Creates embeddings for each sentence and stores them in the FAISS index.
        """
        
        documents = []
        for item in self.data_variable:
            sentence = item.get('sentence', '')
            metadata = item.get('metadata', {})
            # Create a Document object with sentence and metadata
            doc = Document(page_content=sentence, metadata=metadata)
            documents.append(doc)

        # Create the FAISS index from documents using embeddings
        self.index = FAISS.from_documents(documents, self.embeddings)

    def save_index_locally(self) -> bool:
        """
        Saves the current index to a local file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Define the path to save the index
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",'databases', self.index_name)
            if self.index:
                # Save the index locally
                self.index.save_local(save_path)
            else:
                print("No index to save.")
                return False
            return True
        except Exception as e:
            print(f"Exception occurred while saving the index locally: {e}")
            return False

    def load_existing_index(self) -> bool:
        """
        Loads an existing index from a local file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Define the path to load the index
            load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",'databases', self.index_name)
            # Load the index from the local path
            self.index = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
            return True
        except Exception as e:
            print(f"Exception occurred while loading the index: {e}")
            return False

    def query_index(self, query: str, k: int = 5) -> List[Dict]:
        """
        Queries the index with a given query string and returns the top k results.

        Args:
            query (str): The query string.
            k (int): The number of top results to return.

        Returns:
            List[Dict]: List of dictionaries containing the retrieved sentences and metadata.
        """
        if not self.index:
            print("Index is not loaded or created.")
            return []

        # Perform a similarity search on the index
        docs = self.index.similarity_search(query, k=k)
        results = []
        for doc in docs:
            result = {
                'sentence': doc.page_content,
                'metadata': doc.metadata
            }
            results.append(result)
        return results

if __name__ == "__main__":

    from data_preprocessing import Preprocessor
    paper_folder = r"/Users/sduran/Desktop/carpeta sin título"

    preprocessor = Preprocessor(paper_folder)
    preprocessor.enumerate_files()
    preprocessor.process_pdfs()
    data_variable = preprocessor.data

    # Initialize the EmbeddingDBFromData class
    embedding_db = EmbeddingDBFromData(index_name='my_index', data_variable=data_variable)

    # Create embeddings and build the index
    embedding_db.create_embeddings_and_index()

    # Save the index locally
    embedding_db.save_index_locally()

    # Load the index if needed
    embedding_db.load_existing_index()

    # Query the index
    results = embedding_db.query_index(query="Evidence of shorelines on Mars", k=5)
    print(results)