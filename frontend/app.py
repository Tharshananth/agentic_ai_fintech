# finance-agent/app.py

import streamlit as st
from shared import (
    ApiAgentRequest,
    ScrapingAgentRequest,
    RetrieverAgentRequest,
    AnalysisAgentRequest,
    llm_manager
)

st.set_page_config(page_title="Finance Analyst AI", layout="wide")

# Initialize LLM Manager
llm = llm_manager

# Title
st.title("📊 Financial Analyst Assistant")

# Sidebar Input
st.sidebar.header("🔍 User Query")
user_input = st.sidebar.text_area("Enter your financial question or command:")

if st.sidebar.button("Run Analysis"):
    if not user_input.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Processing your request..."):

            # Simulated agent steps (replace these with real API/logic calls)

            # Step 1: API Agent Simulation
            api_data = {
                "region": "US",
                "sector": "Technology",
                "companies": [{"name": "Apple Inc", "symbol": "AAPL"}],
                "stock_data": {
                    "AAPL": {
                        "price": "$190.10",
                        "change": "+1.2%"
                    }
                },
                "status": "success"
            }

            # Step 2: Scraping Agent Simulation
            scraping_data = {
                "nifty_it_summary": {"Total Market Cap": "₹12.5T", "P/E Ratio": "25.6"},
                "market_indices": [
                    {"name": "NIFTY 50", "value": "19,200", "change": "+0.85%"},
                    {"name": "SENSEX", "value": "65,000", "change": "+1.1%"}
                ],
                "status": "success"
            }

            # Step 3: Retriever Agent (LLM on documents)
            doc_query = f"Extract key financial data based on: {user_input}"
            llm_response = llm.llama_index_llm.complete(doc_query)

            # Step 4: Analysis Agent (LLM + data)
            final_analysis = {
                "analysis": "Apple Inc shows strong revenue growth this quarter...",
                "risk_assessment": "Moderate. Market volatility remains a concern.",
                "recommendations": [
                    "Hold Apple stocks until next earnings report.",
                    "Consider diversifying with ETFs in tech sector."
                ],
                "status": "success"
            }

        # Display results
        st.subheader("🧠 AI Financial Analysis")
        st.markdown(f"**Analysis:** {final_analysis['analysis']}")
        st.markdown(f"**Risk Assessment:** {final_analysis['risk_assessment']}")
        st.markdown("**Recommendations:**")
        for rec in final_analysis["recommendations"]:
            st.markdown(f"- {rec}")

        # Display retrieved document insight
        st.subheader("📄 Key Financial Extracts")
        st.markdown(f"**LLM Output:**\n\n{llm_response}")

        # Display stock data
        st.subheader("📈 Stock Market Overview")
        for symbol, data in api_data["stock_data"].items():
            st.markdown(f"**{symbol}**: {data['price']} ({data['change']})")

        # Display market indices
        st.subheader("🏦 Market Indices")
        for index in scraping_data["market_indices"]:
            st.markdown(f"{index['name']}: {index['value']} ({index['change']})")
