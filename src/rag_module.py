import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
import time

# Langchain components
try:
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Langchain components not found. Please install langchain, langchain-community, faiss-gpu, pypdf. Error: {e}")
    FAISS = None
    RecursiveCharacterTextSplitter = None
    DirectoryLoader = None
    PyPDFLoader = None
    TextLoader = None
    HuggingFaceEmbeddings = None
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self, config: Dict):
        """
        Initializes the RAG Manager.

        Args:
            config (Dict): Configuration dictionary containing rag settings:
                - knowledge_base_path (str): Path to the directory containing documents.
                - index_path (str): Path to save/load the FAISS index.
                - embedding_model (str): Name of the sentence-transformer model.
                - chunk_size (int): Size of text chunks.
                - top_k (int): Default number of documents to retrieve.
        """
        if not LANGCHAIN_AVAILABLE:
             logger.error("Langchain or its dependencies are not available. RAGManager cannot function.")
             raise ImportError("Required Langchain components are missing.")

        self.kb_path = Path(config.get("knowledge_base_path", "knowledge_base"))
        self.index_path = Path(config.get("index_path", "models/faiss_index"))
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.chunk_size = config.get("chunk_size", 512)
        self.top_k = config.get("top_k", 3)
        self.embeddings = None
        self.index: Optional[FAISS] = None

        logger.info(f"Initializing RAGManager with KB path: {self.kb_path}, Index path: {self.index_path}")

        try:
            # Initialize embeddings (uses sentence-transformers via HuggingFaceEmbeddings)
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name, model_kwargs={'device': 'cuda'}) # Load on GPU
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load embedding model '{self.embedding_model_name}': {e}")
            # Allow initialization but retrieval will fail later
            return

        # Try to load existing index
        self._load_index()

    def _load_index(self):
        """Loads the FAISS index from the specified path if it exists."""
        if self.index_path.exists() and self.embeddings:
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}...")
                start_time = time.time()
                # allow_dangerous_deserialization is needed for FAISS with langchain > 0.0.200
                self.index = FAISS.load_local(
                    folder_path=str(self.index_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS index loaded successfully in {time.time() - start_time:.2f}s.")
            except Exception as e:
                logger.exception(f"Failed to load FAISS index from {self.index_path}: {e}. Index will need rebuilding.")
                self.index = None
        else:
            logger.warning(f"FAISS index not found at {self.index_path} or embeddings not loaded. Build index first.")
            self.index = None

    def _load_documents(self) -> List:
        """Loads documents from the knowledge base path."""
        logger.info(f"Loading documents from {self.kb_path}...")
        # Define loaders for different file types
        loader_kwargs_txt = {"encoding": "utf8"}
        loader = DirectoryLoader(
            path=str(self.kb_path),
            glob="**/*", # Load all files initially
            loader_cls=TextLoader, # Default loader
            loader_kwargs=loader_kwargs_txt,
            use_multithreading=True,
            show_progress=True,
            silent_errors=True # Skip files it can't load
            # We might need more specific loaders or logic here if handling many types
            # For example, explicitly adding PyPDFLoader:
            # loaders = {
            #     ".pdf": PyPDFLoader,
            #     ".txt": TextLoader,
            # }
            # # Custom loading logic based on file extension...
        )
        try:
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {self.kb_path}.")
            # Filter out empty documents that might result from silent_errors
            documents = [doc for doc in documents if doc.page_content.strip()]
            logger.info(f"{len(documents)} non-empty documents remaining.")
            return documents
        except Exception as e:
            logger.exception(f"Error loading documents from {self.kb_path}: {e}")
            return []

    def _split_documents(self, documents: List) -> List:
        """Splits documents into chunks."""
        logger.info(f"Splitting {len(documents)} documents into chunks of size {self.chunk_size}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.1), # 10% overlap
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks.")
        return split_docs

    def build_index(self, force_rebuild: bool = False):
        """Builds or rebuilds the FAISS index from documents in the knowledge base."""
        if self.index is not None and not force_rebuild:
            logger.info("FAISS index already loaded. Use force_rebuild=True to overwrite.")
            return

        if not self.embeddings:
            logger.error("Embedding model not loaded. Cannot build index.")
            return

        documents = self._load_documents()
        if not documents:
            logger.error("No documents loaded from knowledge base. Cannot build index.")
            return

        split_docs = self._split_documents(documents)
        if not split_docs:
             logger.error("No chunks created after splitting documents. Cannot build index.")
             return

        logger.info(f"Building FAISS index from {len(split_docs)} chunks...")
        start_time = time.time()
        try:
            # Create FAISS index from documents and embeddings
            self.index = FAISS.from_documents(split_docs, self.embeddings)
            logger.info(f"FAISS index built successfully in {time.time() - start_time:.2f}s.")

            # Save the index locally
            self.index_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving FAISS index to {self.index_path}...")
            self.index.save_local(folder_path=str(self.index_path))
            logger.info("FAISS index saved successfully.")

        except Exception as e:
            logger.exception("Failed to build or save FAISS index.")
            self.index = None # Ensure index is None if build fails

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieves relevant document chunks for a given query.

        Args:
            query (str): The user query.
            top_k (int, optional): Number of documents to retrieve. Defaults to value from config.

        Returns:
            List[str]: A list of the page content of the relevant document chunks.
        """
        if self.index is None:
            logger.warning("FAISS index is not loaded. Attempting to load...")
            self._load_index()
            if self.index is None:
                 logger.error("Failed to load FAISS index. Cannot retrieve documents. Please build the index first.")
                 return []

        k = top_k if top_k is not None else self.top_k
        logger.debug(f"Retrieving top {k} documents for query: '{query[:100]}...'")
        start_time = time.time()

        try:
            # Perform similarity search
            retrieved_docs = self.index.similarity_search(query, k=k)
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s.")

            # Extract page content
            results = [doc.page_content for doc in retrieved_docs]
            return results

        except Exception as e:
            logger.exception(f"Error during FAISS similarity search for query: '{query[:100]}...'")
            return []

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration (Mimic loading from config.yaml) ---
    # Note: Assumes this script is run from the project root or paths are adjusted
    example_config = {
        "knowledge_base_path": "knowledge_base",
        "index_path": "models/faiss_index_example", # Use separate index for example
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 256, # Smaller chunk size for quick example
        "top_k": 2
    }

    kb_dir = Path(example_config["knowledge_base_path"])
    index_dir = Path(example_config["index_path"])

    # Create dummy knowledge base if it doesn't exist
    kb_dir.mkdir(exist_ok=True)
    dummy_file_path = kb_dir / "dummy_doc.txt"
    if not dummy_file_path.exists():
        logger.info(f"Creating dummy document: {dummy_file_path}")
        with open(dummy_file_path, "w") as f:
            f.write("This is the first document about Langchain and FAISS. FAISS allows for efficient similarity search.
")
            f.write("The second document discusses vector stores. Vector stores hold embeddings.
")
            f.write("Finally, a third piece of information: Embeddings represent text numerically.")
    # --- End Configuration ---

    if not LANGCHAIN_AVAILABLE:
        logger.error("Cannot run example: Langchain components not installed.")
        sys.exit(1)

    try:
        print("Initializing RAGManager...")
        rag_manager = RAGManager(config=example_config)
        print("RAGManager Initialized.")

        # Build index (force rebuild for example consistency)
        print("\nBuilding RAG index (force_rebuild=True)...")
        rag_manager.build_index(force_rebuild=True)

        if rag_manager.index is None:
             print("\nIndex building failed. Cannot proceed with retrieval.")
             sys.exit(1)

        # Perform retrieval
        test_query = "What are vector stores?"
        print(f"\nRetrieving documents for query: '{test_query}'")
        results = rag_manager.retrieve(query=test_query)

        print(f"\nRetrieved {len(results)} documents:")
        if results:
            for i, doc_content in enumerate(results):
                print(f"--- Document {i+1} ---")
                print(doc_content)
                print("--------------------")
        else:
            print("No relevant documents found.")

    except ImportError as e:
         logger.error(f"Import error during example: {e}. Ensure all RAG dependencies are installed.")
    except Exception as e:
        logger.exception(f"An error occurred during the RAG example run: {e}")

    print("\nRAG module example finished.")

# Required for example
import sys
