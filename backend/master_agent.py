from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    # FORCE 'gemini-1.5-flash' which is the current standard.
    # If this fails, the key itself has permission issues for this model.
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0,
            # Force v1beta which is often required for Flash 1.5 on standard keys
            transport="rest" 
        )
        # Test connection instantly
        llm.invoke("Test connection")
    except Exception as e:
        print(f"Flash failed ({e}), trying Pro...")
        # Fallback to Pro if Flash is region-locked
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            google_api_key=api_key,
            temperature=0
        )

    parser = PydanticOutputParser(pydantic_object=MasterPlan)

    system_prompt = """
    You are the *Master Orchestrator Agent* for a pharmaceutical innovation engine. 
    Your task is to analyze a strategic research query and break it down into a structured JSON plan.
    
    ## Available Worker Agents:
    1. *ClinicalTrialsAgent*
    2. *IQVIAInsightsAgent*
    3. *PatentLandscapeAgent*
    4. *EXIMTrendsAgent*
    5. *WebIntelligenceAgent*
    6. *InternalKnowledgeAgent*

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
