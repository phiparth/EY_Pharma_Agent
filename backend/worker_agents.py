import json
import requests
import random
from langchain_core.tools import tool

# --- ROBUST INITIALIZATION ---
# We wrap the search tool import in a try-except block.
# If the library is missing, the app will not crash; it will simply fallback.
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    ddg_search = DuckDuckGoSearchRun()
    SEARCH_AVAILABLE = True
except ImportError:
    print("WARNING: duckduckgo-search not found. Search tools will return mock data.")
    ddg_search = None
    SEARCH_AVAILABLE = False

# --- 1. Clinical Trials Agent (REAL API) ---
@tool
def clinical_trials_agent(instruction: str, molecule: str, indication: str):
    """
    Queries ClinicalTrials.gov API v2 for real-time study data.
    """
    # For robustness, handle cases where API might be down
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
            
            if not studies:
                return "No active clinical trials found for this specific combination."
            
            formatted_results = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                status = protocol.get("statusModule", {}).get("overallStatus", "Unknown")
                phases = protocol.get("designModule", {}).get("phases", ["N/A"])
                sponsor = protocol.get("identificationModule", {}).get("organization", {}).get("fullName", "Unknown")
                title = protocol.get("identificationModule", {}).get("briefTitle", "")
                nct_id = protocol.get("identificationModule", {}).get("nctId", "")
                
                formatted_results.append(
                    f"- [{nct_id}] {title}\n  Phase: {', '.join(phases)} | Status: {status} | Sponsor: {sponsor}"
                )
            return "\n".join(formatted_results)
        else:
            return f"Error fetching trials: API returned {response.status_code}"
    except Exception as e:
        return f"Connection failed to ClinicalTrials.gov: {str(e)}"

# --- 2. Patent Landscape Agent (Targeted Web Search) ---
@tool
def patent_landscape_agent(instruction: str, molecule: str):
    """
    Searches specialized patent repositories (Google Patents, USPTO) via targeted web queries.
    """
    if not SEARCH_AVAILABLE or not ddg_search:
        return "Search tool unavailable. Please install 'duckduckgo-search'."

    # Targeted query for robustness
    search_query = f"site:patents.google.com OR site:uspto.gov {molecule} patent expiry formulation"
    
    try:
        results = ddg_search.run(search_query)
        return f"Patent Search Results (Source: Google Patents/USPTO via Web):\n{results}"
    except Exception as e:
        return f"Patent search failed: {e}"

# --- 3. IQVIA Insights Agent (Robust Simulator) ---
@tool
def iqvia_insights_agent(instruction: str, therapeutic_area: str):
    """
    Simulates detailed commercial data based on therapeutic area knowledge base.
    """
    market_db = {
        "Oncology": {"size": "180B", "cagr": "12%", "top_competitor": "Keytruda", "trend": "High growth in immunotherapies"},
        "Diabetes": {"size": "60B", "cagr": "4%", "top_competitor": "Ozempic", "trend": "Shift to GLP-1 agonists"},
        "Cardiovascular": {"size": "50B", "cagr": "3.5%", "top_competitor": "Eliquis", "trend": "Stable generic competition"},
        "Rare Disease": {"size": "20B", "cagr": "15%", "top_competitor": "Various Orphan Drugs", "trend": "High value, low volume"}
    }
    
    data = market_db.get(therapeutic_area, {
        "size": f"{random.randint(10, 100)}B", 
        "cagr": f"{random.uniform(2, 10):.1f}%",
        "top_competitor": "Generic Multi-source",
        "trend": "Moderate growth"
    })
    
    return json.dumps({
        "source": "IQVIA_Mock_DB", 
        "therapeutic_area": therapeutic_area,
        "market_size_usd": data["size"],
        "growth_rate_cagr": data["cagr"],
        "key_trend": data["trend"],
        "insight": f"Analysis for {therapeutic_area} suggests {data['trend']} with {data['top_competitor']} leading the market."
    }, indent=2)

# --- 4. EXIM Trends Agent (Robust Simulator) ---
@tool
def exim_trends_agent(instruction: str, molecule: str):
    """
    Simulates API supply chain data.
    """
    countries = ["China", "India", "Germany", "USA"]
    risk_level = random.choice(["Low", "Medium", "High"])
    
    return json.dumps({
        "molecule": molecule,
        "primary_sourcing_hub": random.choice(countries),
        "import_volume_metric_tons": random.randint(50, 500),
        "supply_chain_risk": risk_level,
        "major_exporters": random.sample(countries, 2)
    }, indent=2)

# --- 5. Web Intelligence Agent (Real Search) ---
@tool
def web_intelligence_agent(instruction: str):
    """
    General scientific web search.
    """
    if not SEARCH_AVAILABLE or not ddg_search:
        return "Web search unavailable (Dependency missing)."
        
    try:
        return ddg_search.run(instruction)
    except Exception as e:
        return f"Web search failed: {str(e)}"

# Map string names to functions for the orchestrator
AGENT_MAP = {
    "ClinicalTrialsAgent": clinical_trials_agent,
    "PatentLandscapeAgent": patent_landscape_agent,
    "IQVIAInsightsAgent": iqvia_insights_agent,
    "EXIMTrendsAgent": exim_trends_agent,
    "WebIntelligenceAgent": web_intelligence_agent
}
