from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    # FIX: Use the specific versioned model name
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001", 
        google_api_key=api_key,
        temperature=0
    )

    parser = PydanticOutputParser(pydantic_object=MasterPlan)

    system_prompt = """
    You are the *Master Orchestrator Agent* for a pharmaceutical innovation engine. 
    Your task is to analyze a strategic research query and break it down into a structured JSON plan for your specialized Worker Agents.
    
    ## Available Worker Agents and Capabilities:
    1. *ClinicalTrialsAgent:* Analyzes the competitive pipeline using data from ClinicalTrials.gov.
    2. *IQVIAInsightsAgent:* Provides commercial viability analysis using simulated IQVIA market data.
    3. *PatentLandscapeAgent:* Assesses Intellectual Property (IP) risk using simulated patent filings.
    4. *EXIMTrendsAgent:* Analyzes API export-import trade data for supply chain stability.
    5. *WebIntelligenceAgent:* Performs real-time search simulation for external scientific rationale.
    6. *InternalKnowledgeAgent:* Summarizes internal documents to assess strategic fit.

    ## Task Instructions:
    1. Fill all core research fields (molecule, indication, therapeutic_area).
    2. Select the necessary agents for the required_agents list.
    3. *Crucially*, generate a unique, specific, and clear **System Instruction (prompt)** for every agent listed in required_agents.

    {format_instructions}
    
    User Query: {query}
    """

    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    return chain.invoke({"query": user_query})
