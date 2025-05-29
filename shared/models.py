from pydantic import BaseModel
from typing import List, Dict, Optional

class Company(BaseModel):
    name: str
    symbol: str

class ApiAgentRequest(BaseModel):
    user_input: str

class ApiAgentResponse(BaseModel):
    region: str
    sector: str
    companies: List[Company]
    stock_data: Dict[str, Dict[str, str]]
    status: str

class ScrapingAgentRequest(BaseModel):
    tickers: Optional[List[str]] = None
    include_indices: bool = True

class MarketIndex(BaseModel):
    name: str
    value: str
    change: str

class ScrapingAgentResponse(BaseModel):
    nifty_it_summary: Dict[str, str]
    market_indices: List[MarketIndex]
    status: str

class RetrieverAgentRequest(BaseModel):
    query: str
    documents_path: Optional[str] = None

class RetrieverAgentResponse(BaseModel):
    response: str
    status: str

class AnalysisAgentRequest(BaseModel):
    api_data: ApiAgentResponse
    scraping_data: ScrapingAgentResponse
    retriever_data: RetrieverAgentResponse
    user_query: str

class AnalysisAgentResponse(BaseModel):
    analysis: str
    risk_assessment: str
    recommendations: List[str]
    status: str
