import os
import logging
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Environment Setup ---
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Main function to run the RAG-based question answering pipeline.
    """
    # --- Configuration ---
    PERSIST_DIR = "chroma_db"
    OLLAMA_MODEL = "wangshenzhi/gemma2-9b-chinese-chat"

    # --- Check for Vector Store ---
    if not Path(PERSIST_DIR).exists():
        logging.error(f"Vector store not found at '{PERSIST_DIR}'.")
        logging.error("Please run `python ingest.py` first to create the vector store.")
        return

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
        embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2",
            model_kwargs={'device': device}
        )
        logging.info(f"HuggingFace embeddings initialized successfully on device '{device}' with model 'all-mpnet-base-v2'.")
    except Exception as e:
        logging.error(f"Failed to initialize HuggingFace embeddings: {e}")
        return

    # --- Load Vector Store ---
    logging.info(f"Loading existing vector store from '{PERSIST_DIR}'...")
    try:
        vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        logging.info("Vector store loaded successfully.")
        logging.info("\n--- Vector DB is ready! ---")
        logging.info(f"Total documents in store: {vector_db._collection.count()}")
    except Exception as e:
        logging.error(f"Failed to load vector store: {e}")
        return

    # --- RAG Chain Implementation ---
    logging.info(f"--- Initializing RAG Chain with Ollama model: {OLLAMA_MODEL} ---")

    # 1. Create Retriever
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # Retrieve top 5 most relevant chunks
    )
    logging.info("Retriever created successfully.")

    # 2. Initialize LLM
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        logging.info("Ollama LLM initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Ollama. Is Ollama running? Error: {e}")
        logging.error("Please ensure the Ollama application is running and the specified model is available.")
        logging.error(f"You can pull the model with: ollama pull {OLLAMA_MODEL}")
        return

    # 3. Create Prompt Template
    prompt_template = """
    請根據以下提供的「上下文」來回答「問題」。
    請使用繁體中文來回答。如果根據上下文無法得知答案，請直接說「根據我所擁有的資料，我無法回答這個問題。」，不要試圖編造答案。

    上下文:
    {context}

    問題: {question}

    回答:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    logging.info("Prompt template created.")

    # 4. Create RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG chain created successfully.")

    # --- Interactive Query Loop ---
    print("\n--- QNAP Wiki RAG 系統已就緒 ---")
    print(f"模型: {OLLAMA_MODEL}")
    print("現在您可以開始提問了 (輸入 'exit' 或 'quit' 來結束程式)。\n")

    while True:
        try:
            question = input("請輸入您的問題: ")
            if question.lower() in ["exit", "quit"]:
                print("正在關閉程式...")
                break
            if not question.strip():
                continue

            print("\n正在思考中...")
            
            # Stream the response
            full_response = ""
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

        except KeyboardInterrupt:
            print("\n偵測到中斷指令，正在關閉程式...")
            break
        except Exception as e:
            logging.error(f"\nAn error occurred during query processing: {e}")
            break


if __name__ == "__main__":
    main()
