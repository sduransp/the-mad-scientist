# Importing libraries
import os
import re
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

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
            "api_key":os.getenv("OPENAI_KEY"),
            "api_version":os.getenv("OPENAI_API_VERSION"),
            "tiktoken_model_name": "gpt4-turbo",
            "azure_deployment": "gpt4-turbo",
            "temperature" : 0
        }

        self.client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_KEY")
        )

        self.langchain_client = AzureChatOpenAI(**model_params)

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
        for pdf_file in self.pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            title, authors, year, citation = self.extract_metadata_from_document(documents[0].page_content[0:1000])
            
            for document in documents:
                text = document.page_content
                # Parsear el texto a partir del Abstract o Introducción
                parsed_text = self.parse_from_abstract_or_introduction(text)
                if parsed_text:
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

        # Defining the message text
        template_str = """
            You are an expert in parsing scientific articles. 
            Given the input text which contains the first page of a scientific paper, extract the following information:
            1. The title of the paper.
            2. The authors (in the order they appear).
            3. The year of publication.
            4. The citation in APA format.

            The input text will be the first page of the paper, which contains the title, authors, and publication year. 
            The output should consist of a JSON object containing the following fields:
            - Title: The full title of the paper.
            - Authors: A list of the authors in the format 'Last name, First initial.' (e.g., Smith, J.)
            - Year: The year of publication.
            - Citation: A citation in APA format (e.g., "Smith, J., & Doe, J. (2020). Title of the paper. Journal Name, Volume(Issue), Pages.")

            Return only a valid JSON object.
            Input Text: {document}

            Task:
            Extract the title, authors, year, and generate an APA citation from the provided input text.
            """

        prompt_template = PromptTemplate(
            template = template_str,
            input_variables = ["document"],
            partial_variables={"format_instructions":JsonOutputParser(pydantic_object=PaperProcessor).get_format_instructions()}
        )

        classification_chain = prompt_template | self.langchain_client | JsonOutputParser(pydantic_object = PaperProcessor)

        try:
            output = classification_chain.invoke({"document": document})
        except:
            raise RuntimeError(F"Error while parsing the paper author information")
        
        title, authors, year, citation = output["Title"], output["Authors"], output["Year"], output["Citation"]

        return(title, authors, year, citation)

    def parse_from_abstract_or_introduction(self, text):
        """
        Find Abstract or Introduction and return text from there
        """
        abstract_regex = re.compile(r"(Abstract|Resumen|Introduction|Introducción)", re.IGNORECASE)
        match = abstract_regex.search(text)
        if match:
            return text[match.start():]  
        return text
    def split_into_sentences(self,text):
        """
        Split text into sentences based on typical sentence-ending punctuation (., !, ?) 
        followed by a space and a capital letter or a digit at the start of the next sentence.
        """
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
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
