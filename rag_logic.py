import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class RAGSystem:
    def __init__(self, persist_dir="chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = self._initialize_embeddings()
        self.vector_db = self._load_vector_store()
        self.rag_chain = self._create_rag_chain()

    def _initialize_embeddings(self):
        device_setting = os.getenv("EMBEDDING_DEVICE", "auto").lower()
        device = "cpu"
        if device_setting == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    logging.info("CUDA is available. Automatically selecting NVIDIA GPU for embeddings.")
                elif torch.backends.mps.is_available():
                    device = "mps"
                    logging.info("MPS is available. Automatically selecting Apple GPU for embeddings.")
                else:
                    logging.info("No GPU acceleration available. Using CPU for embeddings.")
            except ImportError:
                logging.warning("PyTorch is not installed, defaulting to CPU. For GPU support, please install PyTorch.")
        elif device_setting in ["cuda", "gpu"]:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    logging.info("Forcing NVIDIA GPU for embeddings based on EMBEDDING_DEVICE setting.")
                elif torch.backends.mps.is_available():
                    device = "mps"
                    logging.info("CUDA not available, using Apple GPU (MPS) for embeddings.")
                else:
                    device = "cpu"
                    logging.warning("No GPU available, falling back to CPU for embeddings.")
            except ImportError:
                device = "cpu"
                logging.warning("PyTorch not available, using CPU for embeddings.")
        elif device_setting == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                    logging.info("Forcing Apple GPU (MPS) for embeddings based on EMBEDDING_DEVICE setting.")
                else:
                    device = "cpu"
                    logging.warning("MPS not available, falling back to CPU for embeddings.")
            except ImportError:
                device = "cpu"
                logging.warning("PyTorch not available, using CPU for embeddings.")
        else:
            logging.info("Forcing CPU for embeddings based on EMBEDDING_DEVICE setting.")

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-mpnet-base-v2",
                model_kwargs={'device': device}
            )
            logging.info(f"HuggingFace embeddings initialized successfully on device '{device}' with model 'all-mpnet-base-v2'.")
            return embeddings
        except Exception as e:
            logging.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise

    def _load_vector_store(self):
        if not Path(self.persist_dir).exists():
            logging.error(f"Vector store not found at '{self.persist_dir}'.")
            logging.error("Please run `python ingest.py` first to create the vector store.")
            raise FileNotFoundError(f"Vector store not found at '{self.persist_dir}'")

        logging.info(f"Loading existing vector store from '{self.persist_dir}'...")
        try:
            vector_db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            logging.info("Vector store loaded successfully.")
            logging.info(f"Total documents in store: {vector_db._collection.count()}")
            return vector_db
        except Exception as e:
            logging.error(f"Failed to load vector store: {e}")
            raise

    def _create_rag_chain(self):
        # This method will now return a retriever factory (a function to create a retriever)
        # instead of a fixed retriever instance.
        def retriever_factory(k: int):
            return self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
            logging.info("Google Gemini LLM initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Google Gemini. Is the API key configured correctly? Error: {e}")
            raise

        prompt_template = '''
        請根據以下提供的「上下文」來回答「問題」。
        請使用繁體中文來回答。如果根據上下文無法得知答案，請直接說「根據我所擁有的資料，我無法回答這個問題。」，不要試圖編造答案。

        上下文:
        {context}

        問題: {question}

        回答:
        '''
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # The RAG chain is now constructed dynamically within the methods that use it.
        # This method will return the components needed to build it.
        return retriever_factory, prompt, llm, format_docs

    def get_answer_stream(self, question: str, k: int = 2):
        retriever_factory, prompt, llm, format_docs = self.rag_chain
        retriever = retriever_factory(k)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.stream(question)

    def get_retrieved_docs(self, question: str, k: int = 2):
        retriever_factory, _, _, _ = self.rag_chain
        retriever = retriever_factory(k)
        return retriever.invoke(question)

