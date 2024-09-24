# Importing libraries
import os
import re
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_store.prompt_manager import PromptManager

class PaperProcessor(BaseModel):
    """
    Model for processing the scientific article information.

    Attributes:
        paper_processed (List): The list containing the title, authors, year and citation
    """

    paper_processed: list = Field("The list containing the title, authors, year and citation.")

class Preprocessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.pdf_files = []
        self.other_files = []
        self.data = []  

        model_params = {
            "azure_endpoint": os.getenv("OPENAI_ENDPOINT"),
            "api_key":os.getenv("NEXT_API_KEY"),
            "api_version":os.getenv("OPENAI_API_VERSION"),
            "tiktoken_model_name": "gpt4-turbo",
            "azure_deployment": "gpt4-turbo",
            "temperature" : 0
        }

        self.client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("NEXT_API_KEY")
        )

        self.langchain_client = AzureChatOpenAI(**model_params)
        self.prompt_manager = PromptManager()

    def enumerate_files(self):
        """
        Listing and classifying files based on its extension.
        """
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    self.pdf_files.append(file_path)
                else:
                    self.other_files.append(file_path)

    def process_pdfs(self):
        """
        Processing PDF files
        """
        # Looping over all pdf files
        print(f"The list contains {len(self.pdf_files)} files")
        for pdf_file in self.pdf_files:
            # Instantiating the pdf loader
            loader = PyPDFLoader(pdf_file)
            # Loading the document
            documents = loader.load()
            # Obtaining metadata about the paper
            title, authors, year, citation = self.extract_metadata_from_document(documents[0].page_content[0:1000])
            # Processing documents in the paper
            for document in documents:
                # Obtaining the text
                text = document.page_content
                # Parsing text from Abstract or introduction
                parsed_text = self.parse_from_abstract_or_introduction(text)
                if parsed_text:
                    # Dividing the text into sentences and attaching the metadata to it
                    self.extract_sentences_and_metadata(parsed_text, title, authors, year, citation)
        

    def extract_metadata_from_document(self, document):
        """
        Extracting metadata from paper
        """
        # Defining variables
        title = None
        authors = []
        year = None
        citation = None

        # Getting prompt
        template_str = self.prompt_manager.get_prompt("document_metadata",0)
        # Creating prompt template
        prompt_template = PromptTemplate(
            template = template_str,
            input_variables = ["document"],
            partial_variables={"format_instructions":JsonOutputParser(pydantic_object=PaperProcessor).get_format_instructions()}
        )
        # Defining the classification chain
        classification_chain = prompt_template | self.langchain_client | JsonOutputParser(pydantic_object = PaperProcessor)

        try:
            # Invoking the classification chain 
            output = classification_chain.invoke({"document": document})
        except Exception as e:
            # Raising runtime error
            print(e)
            raise RuntimeError(F"Error while parsing the paper author information")
        
        # Obtaining information about the paper
        title, authors, year, citation = output["Title"], output["Authors"], output["Year"], output["Citation"]

        return(title, authors, year, citation)

    def parse_from_abstract_or_introduction(self, text):
        """
        Find Abstract or Introduction and return text from there
        """
        # Defining the regex patter and compiling it
        abstract_regex = re.compile(r"(Abstract|Resumen|Introduction|Introducción)", re.IGNORECASE)
        # Evaluating whether it finds a match 
        match = abstract_regex.search(text)
        # If match, return text from the matching section
        if match:
            return text[match.start():]  
        # Otherwise, return full text
        return text
    def split_into_sentences(self,text):
        """
        Split text into sentences based on typical sentence-ending punctuation (., !, ?) 
        followed by a space and a capital letter or a digit at the start of the next sentence.
        """
        # Defining a regex pattern for extracting proper sentences
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        # Spliting text into proper sentences
        sentences = sentence_endings.split(text)
        return sentences
    def extract_sentences_and_metadata(self, text, title, authors, year, citation):
        """
        Split text into sentences and attach metadata
        """
        # Split text
        sentences = self.split_into_sentences(text)  
        # Instantiating phrase counter
        phrase_number = 1
        # Loop over all sentences
        for sentence in sentences:
            sentence = sentence.strip()  
            # If sentence is not empty
            if sentence: 
                # Create metadata structure 
                metadata = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "citation_format_x": citation,
                    "phrase_number": phrase_number
                }
                # Append it to the data structure
                self.data.append({"sentence": sentence, "metadata": metadata})
                phrase_number += 1


    def save_data(self, output_file):
        """
        Guarda los datos procesados en un archivo de salida.
        """
        with open(output_file, 'w') as f:
            for entry in self.data:
                f.write(f"Sentence: {entry['sentence']}\n")
                f.write(f"Metadata: {entry['metadata']}\n")
                f.write("\n")


if __name__ == "__main__":

    paper_folder = r"/Users/sduran/Desktop/carpeta sin título"

    preprocessor = Preprocessor(paper_folder)
    preprocessor.enumerate_files()
    preprocessor.process_pdfs()
    print(preprocessor.data)
