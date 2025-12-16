import json
import requests
import random
from langchain_core.tools import tool

# --- ROBUST SEARCH SETUP ---
SEARCH_AVAILABLE = False
ddg_search = None

try:
    # Method 1: Standard LangChain Wrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    ddg_search = DuckDuckGoSearchRun()
    SEARCH_AVAILABLE = True
except ImportError:
    try:
        # Method 2: Direct Library Fallback
        from duckduckgo_search import DDGS
        class DirectSearch:
            def run(self, query):
                results = DDGS().text(query, max_results=3)
                if results:
                    return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                return "No results found."
        ddg_search = DirectSearch()
        SEARCH_AVAILABLE = True
    except ImportError:
        print("CRITICAL: duckduckgo-search not installed.")
        SEARCH_AVAILABLE = False

# --- 1. Clinical Trials Agent ---
@tool
def clinical_trials_agent(instruction: str, molecule: str, indication: str):
    """Queries ClinicalTrials.gov API v2 for real-time study data."""
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": f"{molecule} {indication}",
        "pageSize": 5,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            studies = data.get("studies", [])
            if not studies: return "No active clinical trials found."
            
            results = []
            for study in studies:
                p = study.get("protocolSection", {})
                status = p.get("statusModule", {}).get("overallStatus", "Unknown")
                phases = p.get("designModule", {}).get("phases", ["N/A"])
                results.append(f"- [{p.get('identificationModule', {}).get('nctId')}] {p.get('identificationModule', {}).get('briefTitle')} ({', '.join(phases)}) - {status}")
            return "\n".join(results)
        return "API Error."
    except Exception as e:
        return f"Connection failed: {e}"

# --- 2. Patent Landscape Agent ---
@tool
def patent_landscape_agent(instruction: str, molecule: str):
    """Searches patents via web."""
    if not SEARCH_AVAILABLE: return "Search unavailable. Install duckduckgo-search."
    return ddg_search.run(f"site:patents.google.com {molecule} patent expiry")

# --- 3. IQVIA Insights Agent (Simulated) ---
@tool
def iqvia_insights_agent(instruction: str, therapeutic_area: str):
    """Simulates market data."""
    # Simulation Logic
    data = {
        "size": f"{random.randint(10, 100)}B", 
        "cagr": f"{random.uniform(2, 10):.1f}%",
        "competitors": ["Pfizer", "Novartis", "Merck"]
    }
    return json.dumps(data)

# --- 4. EXIM Trends Agent (Simulated) ---
@tool
def exim_trends_agent(instruction: str, molecule: str):
    """Simulates export-import data."""
    return json.dumps({
        "molecule": molecule,
        "major_source": random.choice(["China", "India", "Germany"]),
        "risk": random.choice(["Low", "Medium", "High"])
    })

# --- 5. Web Intelligence Agent ---
@tool
def web_intelligence_agent(instruction: str):
    """General web search."""
    if not SEARCH_AVAILABLE: return "Search unavailable."
    return ddg_search.run(instruction)

AGENT_MAP = {
    "ClinicalTrialsAgent": clinical_trials_agent,
    "PatentLandscapeAgent": patent_landscape_agent,
    "IQVIAInsightsAgent": iqvia_insights_agent,
    "EXIMTrendsAgent": exim_trends_agent,
    "WebIntelligenceAgent": web_intelligence_agent
}
