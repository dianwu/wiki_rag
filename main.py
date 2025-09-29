import os
import re
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from dotenv import load_dotenv
from lxml import etree
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Environment Setup ---
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class WikiDataProcessor:
    """
    Encapsulates the entire data processing pipeline for MediaWiki XML dumps.
    """
    WIKI_XML_NAMESPACE = "http://www.mediawiki.org/xml/export-0.11/"

    def __init__(self, data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logging.info(f"WikiDataProcessor initialized for directory: {self.data_dir}")

    def find_xml_file(self) -> Optional[Path]:
        """
        Scans the data directory for .xml files and returns the path to the first one found.
        """
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            logging.error(f"Data directory '{self.data_dir}' not found or is not a directory.")
            return None

        try:
            xml_files = [f for f in self.data_dir.iterdir() if f.is_file() and f.suffix == '.xml']
            if not xml_files:
                logging.warning(f"No XML files found in the '{self.data_dir}' directory.")
                return None
            
            selected_file = xml_files[0]
            logging.info(f"Found and selected data file: {selected_file}")
            return selected_file
        except Exception as e:
            logging.error(f"An error occurred while searching for XML files: {e}")
            return None

    def load_and_parse_xml(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Parses a MediaWiki XML file efficiently and extracts page titles and text.
        """
        logging.info(f"Parsing XML file: {file_path}...")
        pages: List[Dict[str, str]] = []
        ns_map = {'mw': self.WIKI_XML_NAMESPACE}
        tag_name = f"{{{self.WIKI_XML_NAMESPACE}}}page"

        try:
            context = etree.iterparse(str(file_path), events=('end',), tag=tag_name)
            for _, elem in context:
                title_elem = elem.find('mw:title', namespaces=ns_map)
                text_elem = elem.find('mw:revision/mw:text', namespaces=ns_map)
                
                if title_elem is not None and text_elem is not None and text_elem.text:
                    pages.append({
                        'title': title_elem.text,
                        'text': text_elem.text
                    })
                
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

            logging.info(f"Successfully parsed {len(pages)} pages.")
            return pages
        except etree.XMLSyntaxError as e:
            logging.error(f"Error parsing XML file: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred during XML parsing: {e}")
            return []

    @staticmethod
    def clean_mediawiki_text(raw_text: str) -> str:
        """
        Cleans MediaWiki text by removing HTML tags and common MediaWiki markup.
        """
        text = BeautifulSoup(raw_text, 'lxml').get_text()
        
        # Remove file and image links
        text = re.sub(r'\[\[(?:File|Image|文件|圖片):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
        # Remove templates
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        # Remove bold and italic
        text = text.replace("'''", "").replace("''", "")
        # Remove headers
        text = re.sub(r'==+\s*(.*?)\s*==+', r'\1', text)
        # Remove external links, keeping the text
        text = re.sub(r'\[http[^\s]*\s (.*?)(\\s.*?)?\]', r'\1', text)
        # Remove wiki links, keeping the alias if it exists
        text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)
        # Clean up leftover whitespace
        text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
        # Normalize spaces within each line (保留換行)
        text = '\n'.join(re.sub(r' +', ' ', line) for line in text.splitlines())
        # Remove empty lines and trim
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

        return text

    def split_documents(self, pages: List[Dict[str, str]]) -> List[Document]:
        """
        Splits the text of each page into smaller LangChain Document objects (chunks).
        """
        logging.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        
        texts = [page['text'] for page in pages]
        metadatas = [{'source': page['title']} for page in pages]
        
        documents = text_splitter.create_documents(texts, metadatas=metadatas)
        logging.info(f"Created {len(documents)} document chunks.")
        return documents

    def run(self) -> Optional[List[Document]]:
        """
        Executes the full data processing pipeline.
        """
        logging.info("--- Starting Data Processing Pipeline ---")
        
        # 1. Find XML file
        xml_file_path = self.find_xml_file()
        if not xml_file_path:
            return None

        # 2. Load and parse XML
        wiki_pages = self.load_and_parse_xml(xml_file_path)
        if not wiki_pages:
            return None

        # 3. Clean page content
        logging.info("Cleaning page content...")
        cleaned_pages = [
            {'title': page['title'], 'text': self.clean_mediawiki_text(page['text'])}
            for page in wiki_pages
        ]
        
        if cleaned_pages:
            logging.info("--- Example of cleaned content ---")
            logging.info(f"Title: {cleaned_pages[0]['title']}")
            logging.info(f"Cleaned Text Snippet: {cleaned_pages[0]['text'][:200]}...")
            logging.info("---------------------------------")

        # 4. Split documents
        documents = self.split_documents(cleaned_pages)
        
        if documents:
            logging.info("--- Example of a document chunk ---")
            logging.info(documents[0])
            logging.info("------------------------------------")
            
        logging.info("--- Data Processing Pipeline Finished ---")
        return documents


def create_vector_store(documents: List[Document], embeddings, persist_directory: str) -> Chroma:
    """
    Creates and persists a Chroma vector store from the given documents.
    """
    logging.info(f"Creating vector store at '{persist_directory}'...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info("Vector store created successfully.")
    return vector_db


def main():
    """
    Main function to run the data processing and vectorization pipeline.
    """
    # --- Configuration ---
    DATA_DIR = "data"
    PERSIST_DIR = "chroma_db"
    
    # --- Determine Embedding Device ---
    device_setting = os.getenv("EMBEDDING_DEVICE", "auto").lower()
    device = "cpu"  # Default to CPU

    if device_setting == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logging.info("CUDA is available. Automatically selecting GPU for embeddings.")
            else:
                logging.info("CUDA not available. Automatically selecting CPU for embeddings.")
        except ImportError:
            logging.warning("PyTorch is not installed, defaulting to CPU. For GPU support, please install PyTorch with CUDA.")
    elif device_setting in ["cuda", "gpu"]:
        device = "cuda"
        logging.info("Forcing GPU for embeddings based on EMBEDDING_DEVICE setting.")
    else:  # 'cpu' or any other value
        logging.info("Forcing CPU for embeddings based on EMBEDDING_DEVICE setting.")

    # --- Initialize Embeddings ---
    try:
        # Use a local, open-source embedding model.
        # This model will be downloaded automatically on the first run.
        embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2",
            model_kwargs={'device': device}
        )
        logging.info(f"HuggingFace embeddings initialized successfully on device '{device}' with model 'all-mpnet-base-v2'.")
    except Exception as e:
        logging.error(f"Failed to initialize HuggingFace embeddings: {e}")
        return

    # --- Vector Store Handling ---
    vector_db: Optional[Chroma] = None
    if Path(PERSIST_DIR).exists():
        logging.info(f"Loading existing vector store from '{PERSIST_DIR}'...")
        vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        logging.info("Vector store loaded successfully.")
    else:
        logging.info("No existing vector store found. Starting data processing pipeline...")
        processor = WikiDataProcessor(data_dir=DATA_DIR)
        processed_documents = processor.run()
        
        if processed_documents:
            logging.info(f"Data processing successful. Obtained {len(processed_documents)} documents.")
            vector_db = create_vector_store(processed_documents, embeddings, PERSIST_DIR)
        else:
            logging.warning("Data processing failed or produced no documents. Vector store not created.")

    if vector_db:
        logging.info("\n--- Vectorization complete! ---")
        logging.info(f"Vector DB is ready. Total documents in store: {vector_db._collection.count()}")
        # Next step: Implement the RAG chain using this vector_db.
    else:
        logging.warning("\nVectorization failed. Could not create or load the vector store.")


if __name__ == "__main__":
    main()