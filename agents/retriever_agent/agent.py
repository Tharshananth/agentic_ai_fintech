import sys
sys.path.append('/content/finance-agent')

from fastapi import FastAPI, HTTPException
import torch
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from shared.llm_manager import llm_manager
from shared.models import RetrieverAgentRequest, RetrieverAgentResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retriever Agent Service")

class RetrieverAgent:
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.default_docs_path = "/kaggle/input/ass-raga-ai-1/finance-agent-11/agents/retriever_agent/data"  # Default path in Colab
        
    async def process(self, request: RetrieverAgentRequest) -> RetrieverAgentResponse:
        try:
            # Initialize RAG if not already done
            if self.query_engine is None:
                await self._initialize_rag(request.documents_path or self.default_docs_path)
            
            # Query the documents
            response = await self._query_documents(request.query)
            
            return RetrieverAgentResponse(
                response=response,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Retriever Agent error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _initialize_rag(self, documents_path: str):
        """Initialize the RAG system"""
        try:
            logger.info(f"Initializing RAG with documents from: {documents_path}")
            
            # Load documents
            documents = SimpleDirectoryReader(documents_path).load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            # Setup embedding model
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            )
            
            # Create service context with shared LLM
            service_context = ServiceContext.from_defaults(
                chunk_size=1024,
                llm=llm_manager.llama_index_llm,
                embed_model=embed_model
            )
            
            # Create vector index
            self.index = VectorStoreIndex.from_documents(
                documents, 
                service_context=service_context
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine()
            
            logger.info("✅ RAG system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            raise
    
    async def _query_documents(self, query: str) -> str:
        """Query the RAG system"""
        try:
            if self.query_engine is None:
                raise Exception("RAG system not initialized")
                
            response = self.query_engine.query(query)
            return str(response)
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise

# Initialize agent
retriever_agent = RetrieverAgent()

@app.post("/process", response_model=RetrieverAgentResponse)
async def process_request(request: RetrieverAgentRequest):
    return await retriever_agent.process(request)

@app.post("/initialize")
async def initialize_rag(documents_path: str = "/content/documents"):
    """Manually initialize RAG system"""
    try:
        await retriever_agent._initialize_rag(documents_path)
        return {"status": "success", "message": "RAG system initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "Retriever Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
