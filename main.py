import os
import re
import warnings
from lxml import etree
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def find_xml_file(data_dir="data"):
    """
    Scans the specified directory for .xml files and returns the path to the first one found.

    Args:
        data_dir (str): The directory to scan.

    Returns:
        str: The full path to the first XML file found, or None if no XML file is found.
    """
    xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml') and os.path.isfile(os.path.join(data_dir, f))]
    
    if not xml_files:
        print("No XML files found in the 'data' directory.")
        return None
    
    # Automatically select the first file found
    selected_file = xml_files[0]
    full_path = os.path.abspath(os.path.join(data_dir, selected_file))
    
    print(f"Found and selected data file: {full_path}")
    return full_path

def load_and_parse_xml(file_path):
    """
    Parses a MediaWiki XML file efficiently and extracts page titles and text.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary
                    represents a page with 'title' and 'text' keys.
    """
    print("Parsing XML file...")
    pages = []
    # Namespace is important for MediaWiki XML, corrected to 0.11.
    ns_map = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
    
    try:
        context = etree.iterparse(file_path, events=('end',), tag='{http://www.mediawiki.org/xml/export-0.11/}page')
        
        for event, elem in context:
            title_elem = elem.find('mw:title', namespaces=ns_map)
            text_elem = elem.find('mw:revision/mw:text', namespaces=ns_map)
            
            if title_elem is not None and text_elem is not None and text_elem.text:
                pages.append({
                    'title': title_elem.text,
                    'text': text_elem.text
                })
            
            # Clear the element to free up memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        print(f"Successfully parsed {len(pages)} pages.")
        return pages
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file: {e}")
        return []

def clean_mediawiki_text(raw_text):
    """
    Cleans MediaWiki text by removing HTML tags and common MediaWiki markup.

    Args:
        raw_text (str): The raw text content from a MediaWiki page.

    Returns:
        str: The cleaned, plain text.
    """
    # 1. Remove HTML tags
    text = BeautifulSoup(raw_text, 'lxml').get_text()
    
    # 2. Remove MediaWiki file/image links
    text = re.sub(r'\[\[(?:File|Image|文件|圖片):.*?\]\]', '', text, flags=re.IGNORECASE)
    
    # 3. Remove MediaWiki templates
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    
    # 4. Remove bold and italics
    text = text.replace("'''", "").replace("''", "")
    
    # 5. Remove section headers (keep the title)
    text = re.sub(r'==+\s*(.*?)\s*==+', r'\1', text)
    
    # 6. Remove external links (e.g., [http://...])
    text = re.sub(r'\[http[^\s]*\s(.*?)\]', r'\1', text)
    
    # 7. Remove remaining simple wiki links (keep the text)
    text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)
    
    # 8. Clean up leftover whitespace
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    
    return text

def split_documents(pages):
    """
    Splits the text of each page into smaller documents (chunks).

    Args:
        pages (list[dict]): A list of cleaned page dictionaries.

    Returns:
        list[Document]: A flat list of LangChain Document objects.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
    )
    
    texts = [page['text'] for page in pages]
    metadatas = [{'source': page['title']} for page in pages]
    
    documents = text_splitter.create_documents(texts, metadatas=metadatas)
        
    print(f"Created {len(documents)} document chunks.")
    return documents

if __name__ == "__main__":
    xml_file_path = find_xml_file()
    if xml_file_path:
        wiki_pages = load_and_parse_xml(xml_file_path)
        if wiki_pages:
            print("Cleaning page content...")
            # Create a new list for cleaned pages to preserve original data if needed
            cleaned_pages = []
            for page in wiki_pages:
                cleaned_pages.append({
                    'title': page['title'],
                    'text': clean_mediawiki_text(page['text'])
                })
            
            # For debugging, print the title and a snippet of cleaned text from the first page
            if cleaned_pages:
                print("--- Example of cleaned content ---")
                print(f"Title: {cleaned_pages[0]['title']}")
                print(f"Cleaned Text Snippet: {cleaned_pages[0]['text'][:200]}...")
                print("---------------------------------")

            documents = split_documents(cleaned_pages)
            
            if documents:
                print("--- Example of a document chunk ---")
                print(documents[0])
                print("------------------------------------")