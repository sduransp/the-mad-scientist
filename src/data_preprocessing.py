# Importing libraries
import os
import re
import difflib
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from collections import Counter
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
    def remove_headers_and_footers(self, text, all_texts):
        """
        Remove repetitive headers and footers based on common first and last lines across multiple pages.
        """
        # Split the text into lines
        lines = text.split("\n")

        # Get the first and last lines of all pages
        first_lines = [doc.split("\n")[0] for doc in all_texts if doc.split("\n")]
        last_lines = [doc.split("\n")[-1] for doc in all_texts if doc.split("\n")]

        # Find the most common first and last lines (potential headers and footers)
        common_first = Counter(first_lines).most_common(1)[0][0]
        common_last = Counter(last_lines).most_common(1)[0][0]

        # Remove the header if it matches the common first line
        if lines[0].strip() == common_first.strip():
            lines = lines[1:]

        # Remove the footer if it matches the common last line
        if lines[-1].strip() == common_last.strip():
            lines = lines[:-1]

        # Rebuild the cleaned text
        cleaned_text = "\n".join(lines)

        return cleaned_text

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
            print(f"There are {len(documents)} documents")
            # Obtaining metadata about the paper
            title, authors, year, citation = self.extract_metadata_from_document(documents[0].page_content[0:1000])
            # Get the text of all documents for detecting headers and footers
            all_texts = [doc.page_content for doc in documents]
            # Processing documents in the paper
            for document in documents:
                # Obtaining the text
                text = document.page_content
                # Remove headers and footers
                cleaned_text = self.remove_headers_and_footers(text, all_texts)
                # Parsing text from Abstract or introduction
                parsed_text, stop_processing = self.parse_from_abstract_or_introduction(cleaned_text, citation)
                if parsed_text:
                    # Dividing the text into sentences and attaching the metadata to it
                    self.extract_sentences_and_metadata(parsed_text, title, authors, year, citation)
                if stop_processing:
                    print("Stopping further document processing as final section is detected.")
                    break  # Detener el bucle interno de documentos
            
            # # If stop_processing is True, stop processing other PDFs as well
            # if stop_processing:
            #     print("Stopping processing of further PDFs.")
            #     break  # Detener el bucle externo de PDFs

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
    
    # Function to measure similarity between two strings
    def is_own_citation(self, citation, own_citation, threshold):
        # Normalize spaces and compare
        citation_normalized = " ".join(citation.split())
        own_citation_normalized = " ".join(own_citation.split())

        print(f"comparing own citation with: {citation_normalized}")
        
        # Calculate similarity ratio using difflib
        similarity_ratio = difflib.SequenceMatcher(None, citation_normalized, own_citation_normalized).ratio()

        print(f"Similarity ratio: {similarity_ratio}")
        
        return similarity_ratio > threshold

    
    def parse_from_abstract_or_introduction(self, text, own_citation, threshold=0.8):
        """
        Find 'Abstract' or 'Introduction' and return text from there until 'References'.
        Ignore citations that are too similar to the paper's own citation.
        
        Parameters:
        text (str): The text from which to parse.
        own_citation (str): The citation of the own paper that should be ignored.
        threshold (float): A similarity threshold between 0 and 1 to consider a citation as the own citation.
        
        Returns:
        tuple: The parsed text and a flag indicating if processing should stop at a citation.
        """
        # Define regex patterns for the sections 'Abstract', 'Introduction', 'References'
        abstract_regex = re.compile(r"(Abstract|Resumen|Introduction|Introducción)", re.IGNORECASE)
        # Modified references regex to capture spaces and newlines around "References"
        references_regex = re.compile(r"(References|Bibliography|Referencias)", re.IGNORECASE)

        # Find the start of the 'Abstract' or 'Introduction' section
        match_start = abstract_regex.search(text)
        match_end = references_regex.search(text)
        
        # Find the start of the 'References' section
        stop_processing = False
        possible_endings = []

        if match_end:
            possible_endings.append(match_end)
        
        if possible_endings:
            match_end = min(possible_endings, key=lambda x: x.start())
            stop_processing = True

        # If both 'Abstract' or 'Introduction' and 'References' are found
        if match_start and match_end:
            # Return the text between these two sections
            return text[match_start.start():match_end.start()], stop_processing

        # If only the 'Abstract' or 'Introduction' is found, return from there to the end of the text
        if match_start:
            return text[match_start.start():], stop_processing

        # If none are found, return the full text
        return text, stop_processing
    
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
    
    def split_into_paragraphs(self, text):
        """
        Split text into paragraphs based on single or double newlines, which typically 
        indicate the separation between paragraphs in a block of text.
        """
        # Defining a regex pattern that handles both single and double newlines
        paragraph_endings = re.compile(r'\.\s*\n')

        # Split the text where the pattern is found
        paragraphs = paragraph_endings.split(text.strip())

        # Optionally, remove empty paragraphs or trim leading/trailing spaces
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def remove_figure_or_table_paragraphs(self, paragraphs):
        """
        Remove paragraphs that start with references to figures or tables.
        The function will detect patterns such as "Figure 1.", "Table 1.", "Fig. 1.", etc.

        Args:
            paragraphs (List[str]): List of paragraphs to filter.
        
        Returns:
            List[str]: Filtered list of paragraphs without figure or table references.
        """
        # Regular expression to match paragraphs starting with "Figura", "Table", "Figure", or "Fig."
        pattern = re.compile(r'^(Figura|Table|Figure|Fig)\s*\d+\.', re.IGNORECASE)
        
        # Filter out paragraphs that match the pattern
        filtered_paragraphs = [para for para in paragraphs if not pattern.match(para.strip())]
        
        return filtered_paragraphs
    
    def extract_sentences_and_metadata(self, text, title, authors, year, citation):
        """
        Split text into sentences and attach metadata
        """
        # Split text and filter out
        sentences = self.split_into_paragraphs(text) 
        sentences = self.remove_figure_or_table_paragraphs(sentences) 
        
        # Updated citation regex pattern
        citation_regex = re.compile(
            r"^\s*(?:\d+\.\s*)?"  # Optional number and period at the start
            r".*?"                # Any characters (non-greedy)
            r"\(\d{4}\)\.?\s*$",  # Year in parentheses at the end, optional period, end of string
            re.DOTALL             # Dot matches newlines
        )
        
        phrase_number = 1
        # Loop over all sentences
        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            # If sentence is not empty and doesn't match the citation pattern
            if cleaned_sentence and not citation_regex.match(cleaned_sentence):
                # Create metadata structure 
                metadata = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "citation_format_x": citation,
                    "phrase_number": phrase_number
                }
                # Append it to the data structure
                self.data.append({"sentence": cleaned_sentence, "metadata": metadata})
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
