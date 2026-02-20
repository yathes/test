import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai_like import OpenAILike
from qdrant_client import QdrantClient

# --------------------------------------------------
# ENVIRONMENT CONFIG
# --------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8001")

COLLECTION_NAME = "documents"
DATA_DIR = os.getenv("DATA_DIR", "./data")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")

# --------------------------------------------------
# LAZY INITIALIZATION
# --------------------------------------------------
index = None
query_engine = None
qdrant_client = None
vector_store = None
storage_context = None

def initialize_rag():
    """Initialize RAG components on first use"""
    global index, query_engine, qdrant_client, vector_store, storage_context
    
    if qdrant_client is not None:  # Already initialized
        return
    
    try:
        # Set LLM
        Settings.llm = OpenAILike(
            api_base=f"{LLAMA_CPP_URL}/v1",
            api_key="not-needed",
            model="local-model",
            temperature=0.2,
        )
        
        # Connect to Qdrant
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
        )
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=PERSIST_DIR
        )
        
        # Load existing index if available
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            index = load_index_from_storage(storage_context)
            query_engine = index.as_query_engine()
        else:
            index = None
            query_engine = None
            
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG services unavailable: {str(e)}. Make sure Qdrant and llama.cpp are running."
        )

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="Jetson RAG Service")

# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "running"}

# --------------------------------------------------
# INGEST DOCUMENTS FROM /data
# --------------------------------------------------
@app.post("/rag/ingest")
def ingest_documents():
    global index, query_engine
    
    try:
        initialize_rag()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize RAG: {str(e)}"
        )

    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=400, detail="Data folder not found")

    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    index.storage_context.persist()
    query_engine = index.as_query_engine()

    return {
        "status": "success",
        "documents_indexed": len(documents)
    }

# --------------------------------------------------
# QUERY RAG
# --------------------------------------------------
@app.post("/rag/query")
def query_rag(request: QueryRequest):
    
    try:
        initialize_rag()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize RAG: {str(e)}"
        )

    if not query_engine:
        raise HTTPException(
            status_code=400,
            detail="Index not created. Call /rag/ingest first."
        )

    try:
        response = query_engine.query(request.question)
        return {
            "answer": str(response)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
