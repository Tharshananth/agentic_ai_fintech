import sys
sys.path.append("finance-agent")  # Adjust path if needed

from fastapi import FastAPI, HTTPException
import requests
import json
import re
from typing import List
import logging

from shared.llm_manager import llm_manager
from shared.models import ApiAgentRequest, ApiAgentResponse, Company

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Agent Service")

class APIAgent:
    def __init__(self):
        self.YOUR_API_KEY = "B2NEQSJ06TG4PQX0"  # Replace with your actual key

    async def process(self, request: ApiAgentRequest) -> ApiAgentResponse:
        try:
            companies_data = await self._extract_companies(request.user_input)
            stock_data = await self._fetch_stock_data(companies_data["companies"])

            return ApiAgentResponse(
                region=companies_data["region"],
                sector=companies_data["sector"],
                companies=companies_data["companies"],
                stock_data=stock_data,
                status="success"
            )
        except Exception as e:
            logger.error(f"API Agent Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _extract_companies(self, user_input: str) -> dict:
        prompt = f"""
        Given the user question: "{user_input}", answer the following:
        1. Identify the country or region mentioned.
        2. Identify the industry or sector (e.g., Tech, Pharma, Banking, etc.).
        3. List 5-7 relevant companies traded in that region and sector.
        4. For each company, provide the appropriate Alpha Vantage symbol if known (e.g., RELIANCE.BSE).
        Return the output as a JSON object like this:
        {{
          "region": "india",
          "sector": "Technology",
          "companies": [
            {{"name": "Tata Consultancy Services", "symbol": "TCS.BSE"}},
            {{"name": "Infosys", "symbol": "INFY.BSE"}}
          ]
        }}
        Answer:
        """

        response = llm_manager.pipeline(prompt, max_new_tokens=300)[0]['generated_text']

        match = re.search(r"Answer:\s*({.*})", response, re.DOTALL)
        if match:
            json_text = match.group(1)
            try:
                data = json.loads(json_text)
                companies = [Company(**c) for c in data["companies"]]
                return {
                    "region": data["region"],
                    "sector": data["sector"],
                    "companies": companies
                }
            except Exception as e:
                logger.error(f"Failed to parse company data: {e}")
                raise HTTPException(status_code=500, detail="Failed to parse LLM response")
        else:
            raise HTTPException(status_code=500, detail="Could not extract structured data from LLM")

    async def _fetch_stock_data(self, companies: List[Company]) -> dict:
        stock_data = {}
        for company in companies:
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={company.symbol}&apikey={self.YOUR_API_KEY}"
                response = requests.get(url)
                data = response.json()

                if "Time Series (Daily)" in data:
                    latest_date = list(data["Time Series (Daily)"].keys())[0]
                    daily_data = data["Time Series (Daily)"][latest_date]

                    stock_data[company.symbol] = {
                        "company_name": company.name,
                        "date": latest_date,
                        "open": daily_data["1. open"],
                        "high": daily_data["2. high"],
                        "low": daily_data["3. low"],
                        "close": daily_data["4. close"]
                    }
                elif "Note" in data:
                    stock_data[company.symbol] = {"error": "API limit reached"}
                else:
                    stock_data[company.symbol] = {"error": "No data found"}
            except Exception as e:
                logger.error(f"Error fetching stock data for {company.symbol}: {e}")
                stock_data[company.symbol] = {"error": str(e)}
        return stock_data

api_agent = APIAgent()

@app.post("/process", response_model=ApiAgentResponse)
async def process_request(request: ApiAgentRequest):
    return await api_agent.process(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "API Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
