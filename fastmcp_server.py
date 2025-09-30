import logging
from fastmcp import FastMCP
from rag_logic import RAGSystem
import json

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Systems ---
logging.info("Initializing RAG system...")
try:
    rag_system = RAGSystem()
    logging.info("RAG system initialized successfully.")
except Exception as e:
    logging.error(f"Fatal error during RAG system initialization: {e}")
    # Exit if the core system can't be loaded
    exit(1)

# --- Create FastMCP Application ---
mcp = FastMCP(
    title="QNAP Wiki RAG MCP Server",
    description="A FastMCP server providing document retrieval tools for the QNAP Wiki knowledge base.",
    version="2.0.0"
)

@mcp.tool
def retrieve_wiki_documents(question: str, k: int = 10) -> str:
    """ 
    Retrieves relevant document chunks from the QNAP Wiki knowledge base based on a question.
    
    Args:
        question (str): The question to search for.
        k (int): The maximum number of documents to retrieve.
        
    Returns:
        str: A JSON string containing a list of retrieved documents, 
             each with 'page_content' and 'metadata'.
    """
    logging.info(f"Executing 'retrieve_wiki_documents' tool with question: '{question}' and k={k}")
    try:
        # Use the existing RAG logic to get the documents
        retrieved_docs = rag_system.get_retrieved_docs(question, k=k)
        
        # Format the documents into a list of dictionaries
        docs_list = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in retrieved_docs
        ]
        
        # Return the result as a JSON string, as FastMCP tools return strings
        return json.dumps(docs_list, ensure_ascii=False)
    except Exception as e:
        logging.error(f"An error occurred in retrieve_wiki_documents: {e}")
        # Return a JSON-formatted error message
        return json.dumps({"error": str(e)})

# --- Main Execution ---
if __name__ == "__main__":
    # This allows running the server with `python fastmcp_server.py`
    # It will use the default SSE transport on port 8000
    mcp.run()
