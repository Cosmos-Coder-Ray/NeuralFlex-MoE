"""
Retrieval-Augmented Generation system for NeuralFlex-MoE.
Integrates vector databases and retrieval frameworks for enhanced context.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Vector database imports - we support multiple backends
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS as LangchainFAISS
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    from haystack.document_stores import FAISSDocumentStore
    from haystack.nodes import EmbeddingRetriever
    HAS_HAYSTACK = True
except ImportError:
    HAS_HAYSTACK = False


@dataclass
class RetrievalConfig:
    """Configuration for RAG system"""
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db: str = "faiss"  # faiss, chroma, or haystack


class VectorStore:
    """
    Wrapper around different vector database backends.
    Makes it easy to switch between FAISS, ChromaDB, etc.
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.store = None
        self.embeddings = None
        
        # Initialize based on available backends
        if config.vector_db == "chroma" and HAS_CHROMA:
            self._init_chroma()
        elif config.vector_db == "faiss" and HAS_FAISS:
            self._init_faiss()
        elif config.vector_db == "haystack" and HAS_HAYSTACK:
            self._init_haystack()
        else:
            print(f"Warning: {config.vector_db} not available, using in-memory fallback")
            self._init_fallback()
    
    def _init_chroma(self):
        """Initialize ChromaDB - good for persistent storage"""
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chroma"
        ))
        self.store = client.get_or_create_collection(name="neuraflex_docs")
        print("✓ ChromaDB initialized")
    
    def _init_faiss(self):
        """Initialize FAISS - fastest for similarity search"""
        if HAS_LANGCHAIN:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            # We'll create the index when we add documents
            self.store = None
            print("✓ FAISS with LangChain initialized")
        else:
            # Raw FAISS without LangChain
            self.dimension = 384  # MiniLM embedding size
            self.store = faiss.IndexFlatL2(self.dimension)
            print("✓ Raw FAISS initialized")
    
    def _init_haystack(self):
        """Initialize Haystack - great for production pipelines"""
        self.store = FAISSDocumentStore(
            embedding_dim=384,
            faiss_index_factory_str="Flat"
        )
        self.retriever = EmbeddingRetriever(
            document_store=self.store,
            embedding_model=self.config.embedding_model
        )
        print("✓ Haystack initialized")
    
    def _init_fallback(self):
        """Simple in-memory fallback when nothing else is available"""
        self.store = {"documents": [], "embeddings": []}
        print("✓ In-memory store initialized")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store"""
        if not texts:
            return
        
        if HAS_CHROMA and isinstance(self.store, chromadb.Collection):
            # ChromaDB expects specific format
            ids = [f"doc_{i}" for i in range(len(texts))]
            self.store.add(
                documents=texts,
                metadatas=metadatas or [{}] * len(texts),
                ids=ids
            )
        
        elif HAS_LANGCHAIN and self.embeddings:
            # LangChain FAISS
            self.store = LangchainFAISS.from_texts(
                texts, 
                self.embeddings,
                metadatas=metadatas
            )
        
        elif HAS_HAYSTACK and hasattr(self, 'retriever'):
            # Haystack format
            from haystack.schema import Document
            docs = [Document(content=text, meta=meta or {}) 
                   for text, meta in zip(texts, metadatas or [{}]*len(texts))]
            self.store.write_documents(docs)
            self.store.update_embeddings(self.retriever)
        
        else:
            # Fallback - just store the texts
            self.store["documents"].extend(texts)
        
        print(f"Added {len(texts)} documents to vector store")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Search for relevant documents"""
        k = top_k or self.config.top_k
        
        if HAS_CHROMA and isinstance(self.store, chromadb.Collection):
            results = self.store.query(
                query_texts=[query],
                n_results=k
            )
            docs = results['documents'][0]
            distances = results['distances'][0]
            return list(zip(docs, distances))
        
        elif HAS_LANGCHAIN and self.store:
            results = self.store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        
        elif HAS_HAYSTACK and hasattr(self, 'retriever'):
            results = self.retriever.retrieve(query, top_k=k)
            return [(doc.content, doc.score) for doc in results]
        
        else:
            # Fallback - simple keyword matching
            docs = self.store.get("documents", [])
            scores = [self._simple_similarity(query, doc) for doc in docs]
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return ranked[:k]
    
    @staticmethod
    def _simple_similarity(query: str, doc: str) -> float:
        """Basic similarity for fallback"""
        query_words = set(query.lower().split())
        doc_words = set(doc.lower().split())
        if not query_words:
            return 0.0
        return len(query_words & doc_words) / len(query_words)


class RAGPipeline:
    """
    Complete RAG pipeline that combines retrieval with generation.
    This is what you'd actually use in production.
    """
    
    def __init__(self, model, tokenizer, config: Optional[RetrievalConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RetrievalConfig()
        self.vector_store = VectorStore(self.config)
        
        # Text splitter for chunking documents
        if HAS_LANGCHAIN:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        else:
            self.text_splitter = None
    
    def index_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Index a collection of documents for retrieval.
        Splits them into chunks if they're too long.
        """
        if self.text_splitter:
            # Smart chunking with LangChain
            chunks = []
            chunk_metadatas = []
            
            for i, doc in enumerate(documents):
                doc_chunks = self.text_splitter.split_text(doc)
                chunks.extend(doc_chunks)
                
                # Preserve metadata for each chunk
                meta = metadatas[i] if metadatas else {}
                chunk_metadatas.extend([{**meta, "chunk_id": j} 
                                       for j in range(len(doc_chunks))])
            
            self.vector_store.add_documents(chunks, chunk_metadatas)
        else:
            # Simple chunking fallback
            chunks = []
            for doc in documents:
                # Split into fixed-size chunks
                for i in range(0, len(doc), self.config.chunk_size):
                    chunks.append(doc[i:i + self.config.chunk_size])
            
            self.vector_store.add_documents(chunks, metadatas)
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieve relevant context for a query"""
        results = self.vector_store.search(query, top_k)
        
        # Combine retrieved documents into context
        context_parts = []
        for doc, score in results:
            context_parts.append(f"[Relevance: {score:.2f}]\n{doc}")
        
        return "\n\n".join(context_parts)
    
    def generate_with_retrieval(
        self, 
        query: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        use_retrieval: bool = True
    ) -> Dict[str, any]:
        """
        Generate a response using RAG.
        Retrieves relevant context first, then generates.
        """
        # Step 1: Retrieve relevant context
        if use_retrieval:
            context = self.retrieve_context(query)
            
            # Build prompt with context
            prompt = f"""Context information:
{context}

Question: {query}

Answer based on the context above:"""
        else:
            prompt = query
        
        # Step 2: Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "query": query,
            "context": context if use_retrieval else None,
            "response": response,
            "prompt_length": len(inputs.input_ids[0])
        }


def create_rag_system(model, tokenizer, documents: Optional[List[str]] = None):
    """
    Convenience function to set up a complete RAG system.
    
    Usage:
        rag = create_rag_system(model, tokenizer, my_documents)
        result = rag.generate_with_retrieval("What is quantum computing?")
    """
    config = RetrievalConfig()
    rag = RAGPipeline(model, tokenizer, config)
    
    if documents:
        print(f"Indexing {len(documents)} documents...")
        rag.index_documents(documents)
        print("✓ RAG system ready")
    
    return rag
