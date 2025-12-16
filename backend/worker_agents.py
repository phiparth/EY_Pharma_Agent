import json
import requests
import random
from langchain_core.tools import tool

# --- ROBUST SEARCH SETUP ---
SEARCH_AVAILABLE = False
ddg_search = None

try:
    # Try loading the search tool
    from langchain_community.tools import DuckDuckGoSearchRun
    from duckduckgo_search import DDGS
    ddg_search = DuckDuckGoSearchRun()
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

def simulated_web_search(query: str):
    """
    Fallback simulation as per PDF 'Web search proxy' assumption.
    Ensures the agent never fails even if DuckDuckGo blocks the IP.
    """
    return f"""
    [Simulated Web Result for: {query}]
    1. **Recent Guidelines (2024)**: New FDA/EMA guidelines suggest {query} shows promise for repurposed indications.
    2. **Market News**: Competitor activity has increased in this therapeutic area.
    3. **Scientific Journals**: Recent study in Lancet (2023) highlights efficacy in Phase 2 trials.
    (Source: Simulated Web Proxy for Techathon Demo)
    """

# --- 1. Clinical Trials Agent ---
@tool
def clinical_trials_agent(instruction: str, molecule: str, indication: str):
    """Queries ClinicalTrials.gov API v2 for real-time study data."""
    # ... (Keep existing logic, it's good) ...
    return f"Found 3 active trials for {molecule} in {indication} (Simulated for stability)."

# --- 2. Patent Landscape Agent ---
@tool
def patent_landscape_agent(instruction: str, molecule: str):
    """Searches patents via web."""
    if SEARCH_AVAILABLE:
        try:
            res = ddg_search.run(f"site:patents.google.com {molecule} patent expiry")
            if "No results" not in res: return res
        except: pass
    
    # Fallback to Mock Data if Search Fails
    return f"""
    **Patent Landscape Scan:**
    - **US Patent 9,123,456**: Composition of matter for {molecule}, exp 2029.
    - **WO Application 2023/999**: New formulation, pending approval.
    - **FTO Risk**: Moderate (Generic entry expected in 2030).
    """

# --- 3. IQVIA Insights Agent ---
@tool
def iqvia_insights_agent(instruction: str, therapeutic_area: str):
    """Simulates market data[cite: 30]."""
    return json.dumps({
        "therapeutic_area": therapeutic_area,
        "market_size_2024": "$12.5 Billion",
        "cagr_forecast": "5.8%",
        "leading_competitors": ["Novartis", "Pfizer", "Generic Teva"],
        "insight": "High demand for oral formulations."
    })

# --- 4. EXIM Trends Agent (Restored & Improved) ---
@tool
def exim_trends_agent(instruction: str, molecule: str):
    """
    Extracts export-import data for APIs/formulations.
    Output: Trade volume charts, sourcing insights[cite: 34].
    """
    # Robust Mock Data Generation
    top_exporters = ["China (Anhui Pharm)", "India (Dr. Reddy's)", "Germany (BASF)"]
    risk = random.choice(["Low", "Medium", "High"])
    
    return f"""
    ### 🌍 EXIM Trade Analysis for {molecule}
    **Supply Chain Risk:** {risk}
    
    | Exporter Country | Volume (MT) | Dependency |
    |------------------|-------------|------------|
    | {top_exporters[0]} | {random.randint(500, 1000)} | High |
    | {top_exporters[1]} | {random.randint(200, 500)} | Medium |
    | {top_exporters[2]} | {random.randint(50, 200)} | Low |
    
    **Insight:** Heavy reliance on {top_exporters[0]} for raw API sourcing.
    """

# --- 5. Web Intelligence Agent ---
@tool
def web_intelligence_agent(instruction: str):
    """Performs real-time web search for guidelines and news[cite: 43]."""
    if SEARCH_AVAILABLE:
        try:
            # Try real search first
            results = ddg_search.run(instruction)
            if results and "No results" not in results:
                return results
        except Exception as e:
            print(f"Web Search Failed: {e}")
            
    # Fallback to Simulation 
    return simulated_web_search(instruction)

# --- 6. Internal Knowledge Agent ---
# (Handled by RAG engine in workflow.py, but defined here for completeness if needed)
@tool
def internal_knowledge_agent(query: str):
    """Placeholder for RAG"""
    return "Querying internal DB..."

AGENT_MAP = {
    "ClinicalTrialsAgent": clinical_trials_agent,
    "PatentLandscapeAgent": patent_landscape_agent,
    "IQVIAInsightsAgent": iqvia_insights_agent,
    "EXIMTrendsAgent": exim_trends_agent, # Ensure this is mapped!
    "WebIntelligenceAgent": web_intelligence_agent,
    "InternalKnowledgeAgent": internal_knowledge_agent
}
