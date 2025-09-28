import os
import re
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from lxml import etree
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def main():
    """
    Main function to run the data processing.
    """
    processor = WikiDataProcessor(data_dir="data")
    processed_documents = processor.run()
    
    if processed_documents:
        logging.info(f"\nSuccessfully processed the data and obtained {len(processed_documents)} documents.")
        # The subsequent phases (vectorization, RAG chain) will start here.
    else:
        logging.warning("\nData processing failed or produced no documents.")


if __name__ == "__main__":
    main()