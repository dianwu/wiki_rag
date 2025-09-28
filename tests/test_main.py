import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Make sure main.py is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import WikiDataProcessor

# --- Mock Data ---
MOCK_WIKI_XML = (
    f'<mediawiki xmlns="{WikiDataProcessor.WIKI_XML_NAMESPACE}" xml:lang="en">'
    '  <page>'
    '    <title>Test Page 1</title>'
    '    <revision>'
    '      <text xml:space="preserve">'
    "        This is '''bold text'''.\n"
    "        This is ''italic text''.\n"
    '        == Section 1 ==\n'
    '        Here is a link [[Page 2]].\n'
    '        And a file [[File:Test.jpg|thumb|A test image]].\n'
    '        And a link with alias [[Link|Alias]].\n'
    '        {{Template:Test}}'
    '      </text>'
    '    </revision>'
    '  </page>'
    '  <page>'
    '    <title>Test Page 2</title>'
    '    <revision>'
    '      <text xml:space="preserve">Some more text here. &lt;br /&gt; with a line break.</text>'
    '    </revision>'
    '  </page>'
    '</mediawiki>'
).encode('utf-8')

# --- Test Setup ---

@pytest.fixture
def processor():
    """Provides a WikiDataProcessor instance for testing."""
    return WikiDataProcessor(data_dir="./dummy_data")

# --- Unit Tests ---

def test_clean_mediawiki_text(processor):
    """Tests the static method for cleaning MediaWiki markup."""
    raw_text = "== Title ==\nThis is '''bold''' and ''italic''. [[File:img.png]] [[Link|Alias]]. [[Page]]. {{template}}"
    expected_text = "Title\nThis is bold and italic. Alias. Page."
    assert processor.clean_mediawiki_text(raw_text) == expected_text

    raw_text_html = "Text with <b>HTML</b> tag."
    expected_text_html = "Text with HTML tag."
    assert processor.clean_mediawiki_text(raw_text_html) == expected_text_html
    
    raw_text_empty = " \n \n "
    expected_empty = ""
    assert processor.clean_mediawiki_text(raw_text_empty) == expected_empty

def test_load_and_parse_xml(processor, tmp_path):
    """Tests the XML parsing function by reading from a temporary file."""
    xml_file = tmp_path / "test.xml"
    xml_file.write_bytes(MOCK_WIKI_XML)

    pages = processor.load_and_parse_xml(xml_file)
    
    assert len(pages) == 2
    assert pages[0]['title'] == "Test Page 1"
    assert "'''bold text'''" in pages[0]['text']
    assert pages[1]['title'] == "Test Page 2"
    assert "line break" in pages[1]['text']

def test_split_documents(processor):
    """Tests the document splitting and metadata assignment."""
    pages = [
        {'title': 'Page 1', 'text': 'This is the first sentence. ' * 200},
        {'title': 'Page 2', 'text': 'This is the second sentence.'}
    ]
    documents = processor.split_documents(pages)
    
    assert len(documents) > 1 
    assert documents[0].metadata['source'] == 'Page 1'
    doc_page_2 = [doc for doc in documents if doc.metadata['source'] == 'Page 2']
    assert len(doc_page_2) == 1
    assert doc_page_2[0].page_content == 'This is the second sentence.'

@patch('main.Path.iterdir')
@patch('main.Path.is_file')
@patch('main.Path.is_dir')
@patch('main.Path.exists')
def test_find_xml_file(mock_exists, mock_isdir, mock_isfile, mock_iterdir, processor):
    """Tests the XML file finding logic using pathlib."""
    mock_exists.return_value = True
    mock_isdir.return_value = True
    
    # Test case 1: XML file is found
    mock_iterdir.return_value = [Path("dummy_data/test.xml"), Path("dummy_data/other.txt")]
    mock_isfile.return_value = True
    assert processor.find_xml_file() == Path("dummy_data/test.xml")

    # Test case 2: No XML files in the directory
    mock_iterdir.return_value = [Path("dummy_data/no.txt"), Path("dummy_data/another.log")]
    assert processor.find_xml_file() is None

    # Test case 3: Directory does not exist
    mock_exists.return_value = False
    assert processor.find_xml_file() is None

# --- Integration Test ---

def test_run_pipeline_integration(tmp_path):
    """
    An integration test for the full data processing pipeline.
    It creates a real temporary XML file and runs the processor on it.
    """
    # 1. Setup: Create a temporary data directory and a mock XML file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    xml_file = data_dir / "wiki.xml"
    xml_file.write_bytes(MOCK_WIKI_XML)

    # 2. Execution: Run the processor on the temporary directory
    processor = WikiDataProcessor(data_dir=str(data_dir), chunk_size=150, chunk_overlap=20)
    documents = processor.run()

    # 3. Verification: Check the results
    assert documents is not None
    assert len(documents) > 0

    # Check document content and metadata
    page_1_found = False
    page_2_found = False
    for doc in documents:
        assert 'source' in doc.metadata
        if doc.metadata['source'] == 'Test Page 1':
            page_1_found = True
            # Check if cleaning was successful
            assert "'''bold text'''" not in doc.page_content
            assert "bold text" in doc.page_content
            assert "Alias" in doc.page_content
            assert "Page 2" in doc.page_content
        if doc.metadata['source'] == 'Test Page 2':
            page_2_found = True
            assert "Some more text here" in doc.page_content

    assert page_1_found, "Document chunk for 'Test Page 1' not found."
    assert page_2_found, "Document chunk for 'Test Page 2' not found."