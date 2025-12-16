from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def get_fallback_llm(api_key: str):
    """
    Tries multiple model names until one works.
    Fixes the '404 Not Found' error by falling back to older/stable models.
    """
    # Priority list of models to try
    models_to_try = [
        "gemini-1.5-flash",       # Newest, fast
        "gemini-1.5-flash-001",   # Specific version
        "gemini-1.5-pro",         # Powerful
        "gemini-pro"              # Oldest, most stable fallback
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=api_key,
                temperature=0
            )
            # Test invocation to ensure it actually connects
            llm.invoke("test")
            print(f"DEBUG: Successfully connected to {model_name}")
            return llm
        except Exception as e:
            print(f"WARNING: {model_name} failed. Trying next... ({e})")
            last_error = e
            continue
            
    raise ValueError(f"All Gemini models failed. Please check API Key. Last Error: {last_error}")

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    # Use the fallback logic to get a working LLM
    llm = get_fallback_llm(api_key)

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
