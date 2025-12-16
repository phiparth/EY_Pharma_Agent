from langchain_openai import ChatOpenAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    # We use gpt-4o-mini as it is cheap, fast, and great for planning
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        api_key=api_key,
        temperature=0
    )

    parser = PydanticOutputParser(pydantic_object=MasterPlan)

    system_prompt = """
    You are the *Master Orchestrator Agent* for a pharmaceutical innovation engine. 
    Your task is to analyze a strategic research query and break it down into a structured JSON plan.
    
    ## Available Worker Agents:
    1. *ClinicalTrialsAgent*: Checks ClinicalTrials.gov for active studies.
    2. *IQVIAInsightsAgent*: Simulates commercial market data.
    3. *PatentLandscapeAgent*: Checks patent expiration and IP risks.
    4. *EXIMTrendsAgent*: Checks API supply chain data.
    5. *WebIntelligenceAgent*: General scientific web search.
    6. *InternalKnowledgeAgent*: RAG search on uploaded PDFs.

    ## Task Instructions:
    1. Fill core research fields (molecule, indication, therapeutic_area).
    2. Select necessary agents.
    3. Generate a unique, specific System Instruction for every agent.

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
