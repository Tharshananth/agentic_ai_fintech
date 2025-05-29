import sys
sys.path.append('/content/finance-agent')

from fastapi import FastAPI, HTTPException
import requests
from bs4 import BeautifulSoup
import os
from shared.models import ScrapingAgentRequest, ScrapingAgentResponse, MarketIndex
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Scraping Agent Service")

class ScrapingAgent:
    def __init__(self):
        self.default_tickers = {
            "NIFTY_50": "NIFTY_50:INDEXNSE",
            "SENSEX": "SENSEX:INDEXBOM", 
            "NIFTY_BANK": "NIFTY_BANK:INDEXNSE",
            "BSE-MIDCAP": "BSE_MIDCAP:INDEXBOM",
            "NIFTY_NEXT_50": "NIFTY_NEXT_50:INDEXNSE",
            "NIFTY_500": "NIFTY_500:INDEXNSE",
            "NIFTY_MID_LIQ_15": "NIFTY_MID_LIQ_15:INDEXNSE"
        }
        
    async def process(self, request: ScrapingAgentRequest) -> ScrapingAgentResponse:
        try:
            # Step 1: Scrape Nifty IT summary
            nifty_it_summary = await self._scrape_nifty_it()
            
            # Step 2: Scrape market indices
            market_indices = await self._scrape_market_indices()
            
            return ScrapingAgentResponse(
                nifty_it_summary=nifty_it_summary,
                market_indices=market_indices,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Scraping Agent error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _scrape_nifty_it(self) -> dict:
        """Scrape Nifty IT index data"""
        try:
            ticker = 'NIFTY_IT'
            url = f"https://www.google.com/finance/quote/{ticker}:INDEXNSE"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            summary_data = {}
            # Extract summary data
            for item in soup.select('.gyFHrc'):
                try:
                    key = item.select_one('.mfs7Fc').text.strip()
                    value = item.select_one('.P6K39c').text.strip()
                    summary_data[key] = value
                except AttributeError:
                    continue
                    
            return summary_data
            
        except Exception as e:
            logger.error(f"Error scraping Nifty IT: {str(e)}")
            return {"error": str(e)}
    
    async def _scrape_market_indices(self) -> list:
        """Scrape major market indices"""
        results = []
        
        for name, ticker_code in self.default_tickers.items():
            try:
                url = f"https://www.google.com/finance/quote/{ticker_code}"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                try:
                    value = soup.select_one('.YMlKec.fxKbKc').text
                    change = soup.select_one('.JwB6zf').text
                    results.append(MarketIndex(name=name, value=value, change=change))
                except AttributeError:
                    logger.warning(f"Could not find data for {name}")
                    results.append(MarketIndex(name=name, value="N/A", change="N/A"))
                    
            except Exception as e:
                logger.error(f"Error scraping {name}: {str(e)}")
                results.append(MarketIndex(name=name, value="Error", change=str(e)))
        
        return results

# Initialize agent
scraping_agent = ScrapingAgent()

@app.post("/process", response_model=ScrapingAgentResponse)
async def process_request(request: ScrapingAgentRequest):
    return await scraping_agent.process(request)

@app.get("/health") 
async def health_check():
    return {"status": "healthy", "agent": "Scraping Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
