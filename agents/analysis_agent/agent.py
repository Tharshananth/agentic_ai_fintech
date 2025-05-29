import sys
sys.path.append('/content/finance-agent')

from fastapi import FastAPI, HTTPException
from shared.llm_manager import llm_manager
from shared.models import AnalysisAgentRequest, AnalysisAgentResponse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Analysis Agent Service")

class AnalysisAgent:
    def __init__(self):
        pass

    async def process(self, request: AnalysisAgentRequest) -> AnalysisAgentResponse:
        try:
            # Combine all data sources
            combined_data = await self._combine_data_sources(request)

            # Generate comprehensive analysis
            analysis = await self._generate_analysis(combined_data)

            # Extract risk assessment
            risk_assessment = await self._assess_risk(combined_data)

            # Generate recommendations
            recommendations = await self._generate_recommendations(combined_data)

            return AnalysisAgentResponse(
                analysis=analysis,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                status="success"
            )

        except Exception as e:
            logger.error(f"Analysis Agent error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _combine_data_sources(self, request: AnalysisAgentRequest) -> dict:
        """Combine data from all agents with proper error handling"""
        try:
            # Safely extract API data
            api_data = {}
            if hasattr(request, 'api_data') and request.api_data:
                api_data = {
                    "region": getattr(request.api_data, 'region', 'N/A'),
                    "sector": getattr(request.api_data, 'sector', 'N/A'),
                    "companies": [],
                    "stock_data": getattr(request.api_data, 'stock_data', {})
                }
                
                # Safely extract companies
                if hasattr(request.api_data, 'companies') and request.api_data.companies:
                    api_data["companies"] = [
                        {
                            "name": getattr(c, 'name', 'Unknown'), 
                            "symbol": getattr(c, 'symbol', 'N/A')
                        } 
                        for c in request.api_data.companies
                    ]

            # Safely extract market data
            market_data = {}
            if hasattr(request, 'scraping_data') and request.scraping_data:
                market_data = {
                    "nifty_it_summary": getattr(request.scraping_data, 'nifty_it_summary', 'N/A'),
                    "market_indices": []
                }
                
                # Safely extract market indices
                if hasattr(request.scraping_data, 'market_indices') and request.scraping_data.market_indices:
                    market_data["market_indices"] = [
                        {
                            "name": getattr(idx, 'name', 'Unknown'),
                            "value": getattr(idx, 'value', 0),
                            "change": getattr(idx, 'change', 0)
                        }
                        for idx in request.scraping_data.market_indices
                    ]

            # Safely extract document insights
            document_insights = "No document insights available"
            if hasattr(request, 'retriever_data') and request.retriever_data:
                document_insights = getattr(request.retriever_data, 'response', 'No document insights available')

            return {
                "api_data": api_data,
                "market_data": market_data,
                "document_insights": document_insights,
                "user_query": getattr(request, 'user_query', 'No query provided')
            }
            
        except Exception as e:
            logger.error(f"Error combining data sources: {str(e)}")
            # Return minimal structure to prevent downstream failures
            return {
                "api_data": {},
                "market_data": {},
                "document_insights": "Error extracting document insights",
                "user_query": getattr(request, 'user_query', 'No query provided')
            }

    async def _generate_analysis(self, combined_data: dict) -> str:
        """Generate detailed market and company analysis using LLM"""
        try:
            # Convert complex objects to strings safely
            api_data_str = json.dumps(combined_data.get('api_data', {}), indent=2, default=str)
            market_data_str = json.dumps(combined_data.get('market_data', {}), indent=2, default=str)
            
            prompt = f"""
You are a financial analyst. Based on the following data, provide a detailed analysis for the user's query:

### User Query:
{combined_data.get('user_query', 'No query provided')}

### API Data:
{api_data_str}

### Market Data:
{market_data_str}

### Document Insights:
{combined_data.get('document_insights', 'No insights available')}

Give a concise, professional financial analysis.
"""
            response = llm_manager.openai_chat_llm.predict(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return f"Unable to generate analysis due to error: {str(e)}"

    async def _assess_risk(self, combined_data: dict) -> str:
        """Use LLM to assess risks from the combined data"""
        try:
            # Safely serialize data
            combined_data_str = json.dumps(combined_data, indent=2, default=str)
            
            prompt = f"""
Given the following combined data, identify any major financial or market risks relevant to the user's query.

### Combined Data:
{combined_data_str}

Focus on economic trends, market volatility, company performance, and regulatory risks.
"""
            return llm_manager.openai_chat_llm.predict(prompt)
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            return f"Unable to assess risk due to error: {str(e)}"

    async def _generate_recommendations(self, combined_data: dict) -> str:
        """Generate financial or investment recommendations based on the analysis"""
        try:
            # Safely serialize data
            combined_data_str = json.dumps(combined_data, indent=2, default=str)
            
            prompt = f"""
Using the following combined data, provide strategic recommendations for the user's financial query.

### Combined Data:
{combined_data_str}

Offer practical, realistic, and data-driven suggestions.
"""
            return llm_manager.openai_chat_llm.predict(prompt)
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return f"Unable to generate recommendations due to error: {str(e)}"

# Initialize agent
analysis_agent = AnalysisAgent()

@app.post("/process", response_model=AnalysisAgentResponse)
async def process_request(request: AnalysisAgentRequest):
    return await analysis_agent.process(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "Analysis Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)