# Agentic AI Fintech

A multi-agent orchestration system for financial query analysis using LangGraph and LangChain agents.

## VISUAL FLOW CHART (Markdown Format)
```bash
[ Voice Agent (STT) ]
           ↓
[ API Agent ] ←─── Ticker Input
           ↓
[ Scraping Agent ] ←─── IR URLs
           ↓
[ Retriever Agent (RAG) ]
           ↓
[ Analysis Agent (pandas/logic) ]
           ↓
[ Language Agent (LLM) ]
           ↓
[ Voice Agent (TTS) ]
```

# Agent-by-Agent Explanation
## LLM
```python
model_name = "meta-llama/Llama-2-7b-chat-hf"
```
## STT
 ```python
asr_pipeline = pipeline('automatic-speech-recognition', model='openai/whisper-medium', return_timestamps=True, device=0)
result = asr_pipeline("/kaggle/working/audio.mp3")
```
##  API Agent
   Fetch real-time and historical financial data.

   Alpha Vantage API ( getting the stock related data visit there website to access API Key)

## Scraping Agent
    
   Crawl external websites or documents for earnings reports, news, or filings.

### Tool
  requests, BeautifulSoup

## Retriever Agent
  This is where RAG (Retrieval-Augmented Generation) comes in.
```python
 Setup embedding model
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            )
```
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
```

## Analysis Agent 
  combines all the information  from other agents  and genrates the output 


## 🚀 Quick Start

Follow these steps to get started with the project.

### 1. Clone the Repository

```bash
git clone https://github.com/Tharshananth/agentic_ai_fintech
cd agentic_ai_fintech
```
## Check CUDA Version
```bash
!nvcc --version

```
## Downgrade CUDA (if version >= 12.5)
If your CUDA version is 12.5 or greater, downgrade to 12.4:



```bash
!apt-get remove --purge cuda-* -y
!apt-get autoremove -y
!apt-get autoclean -y
!rm -rf /usr/local/cuda*
---------------------------------------------------------------------------------------------
!wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
!chmod +x cuda_12.4.0_550.54.14_linux.run
!sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit

```

##  Install Python Dependencies
```bash
pip install -r requirements.txt

```
Ensure you're in the correct directory:

```bash
%cd /agentic_ai_fintech/
```
##  Install Specific Versions (Important)
```bash
pip install \
  langchain==0.1.13 \
  langchain-core==0.1.36 \
  langchain-community==0.0.29 \
  langsmith==0.1.38 \
  langchain-openai==0.1.1 \
  langchain-text-splitters==0.0.1 \
  langgraph==0.0.30
```
You can verify versions using:

```bash
pip show langgraph
pip show langchain
...
```

## Run the Orchestrator
```bash
python3 orchestrator/main.py
```
## Start the Agents Individually
In separate terminals or background jobs, run the following:

```bash
python -m uvicorn agents.api_agent:app --port 8001
python -m uvicorn agents.scraping_agent:app --port 8002
python -m uvicorn agents.retriever_agent:app --port 8003
python -m uvicorn agents.analysis_agent:app --port 8004
```

## Health Check
```bash
curl http://localhost:8001/health
```
## project structure
```bash
agentic_ai_fintech/
├── agents/
│   ├── api_agent.py
│   ├── scraping_agent.py
│   ├── retriever_agent.py
│   └── analysis_agent.py
├── orchestrator/
│   └── main.py
├── shared/
│   └── models.py
├── requirements.txt
└── README.md
```



