import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from shared.models import AnalysisAgentRequest, AnalysisAgentResponse
from shared.voice_interface import record_audio, transcribe_audio, speak
import httpx
import logging
from typing import TypedDict, Literal

# Updated LangGraph imports
from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Finance Agent Orchestrator with LangGraph")

# Define the state structure
class GraphState(TypedDict):
    user_query: str
    api_data: dict
    scraping_data: dict
    retriever_data: dict
    analysis_result: dict

# Agent nodes
async def api_agent_node(state: GraphState) -> GraphState:
    user_query = state.get('user_query', '')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8001/process", json={"user_query": user_query})
            response.raise_for_status()
            state['api_data'] = response.json()
    except Exception as e:
        logger.error(f"API Agent call failed: {e}")
        state['api_data'] = {}
    return state

async def scraping_agent_node(state: GraphState) -> GraphState:
    user_query = state.get('user_query', '')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8002/process", json={"user_query": user_query})
            response.raise_for_status()
            state['scraping_data'] = response.json()
    except Exception as e:
        logger.error(f"Scraping Agent call failed: {e}")
        state['scraping_data'] = {}
    return state

async def retriever_agent_node(state: GraphState) -> GraphState:
    user_query = state.get('user_query', '')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8003/process", json={"user_query": user_query})
            response.raise_for_status()
            state['retriever_data'] = response.json()
    except Exception as e:
        logger.error(f"Retriever Agent call failed: {e}")
        state['retriever_data'] = {}
    return state

async def analysis_agent_node(state: GraphState) -> GraphState:
    try:
        payload = {
            "user_query": state.get("user_query", ""),
            "api_data": state.get("api_data", {}),
            "scraping_data": state.get("scraping_data", {}),
            "retriever_data": state.get("retriever_data", {}),
        }
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8004/process", json=payload)
            response.raise_for_status()
            state['analysis_result'] = response.json()
    except Exception as e:
        logger.error(f"Analysis Agent call failed: {e}")
        state['analysis_result'] = {"error": str(e)}
    return state

# Workflow builder
def create_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("api_agent", api_agent_node)
    workflow.add_node("scraping_agent", scraping_agent_node)
    workflow.add_node("retriever_agent", retriever_agent_node)
    workflow.add_node("analysis_agent", analysis_agent_node)

    def route_to_agents(state: GraphState):
        return ["api_agent", "scraping_agent", "retriever_agent"]

    workflow.add_conditional_edges(START, route_to_agents, ["api_agent", "scraping_agent", "retriever_agent"])
    workflow.add_edge("api_agent", "analysis_agent")
    workflow.add_edge("scraping_agent", "analysis_agent")
    workflow.add_edge("retriever_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", END)

    return workflow.compile()

finance_workflow = create_workflow()

@app.post("/process", response_model=AnalysisAgentResponse)
async def process(request: AnalysisAgentRequest):
    initial_state = {
        "user_query": request.user_query,
        "api_data": {},
        "scraping_data": {},
        "retriever_data": {},
        "analysis_result": {}
    }
    try:
        result_state = await finance_workflow.ainvoke(initial_state)
        analysis_result = result_state.get("analysis_result", {})
        if isinstance(analysis_result, dict) and "error" not in analysis_result:
            return analysis_result
        else:
            raise HTTPException(status_code=500, detail=analysis_result.get("error", "Unknown error"))
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Finance Agent Orchestrator"}

@app.get("/workflow/graph")
async def get_workflow_graph():
    try:
        return {
            "nodes": ["api_agent", "scraping_agent", "retriever_agent", "analysis_agent"],
            "edges": [
                "START -> [api_agent, scraping_agent, retriever_agent]",
                "api_agent -> analysis_agent",
                "scraping_agent -> analysis_agent",
                "retriever_agent -> analysis_agent",
                "analysis_agent -> END"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔊 CLI mode for voice interaction
if __name__ == "__main__":
    import asyncio
    import uvicorn

    async def voice_mode():
        print("🎤 Starting voice interaction mode...")
        audio_path = record_audio(duration=6)
        user_text = transcribe_audio(audio_path)
        print("🧠 You said:", user_text)

        request = AnalysisAgentRequest(user_query=user_text)
        result = await process(request)
        print("📊 Analysis Result:", result)
        speak(result.get("response", "No response generated."))

    if "--voice" in sys.argv:
        asyncio.run(voice_mode())
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
