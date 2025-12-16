import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def get_available_model(api_key: str):
    """
    Asks Google which models are available for this Key and picks the best one.
    """
    genai.configure(api_key=api_key)
    try:
        # 1. List all models your key can access
        all_models = list(genai.list_models())
        
        # 2. Filter for text-generation models
        valid_models = [
            m.name for m in all_models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        print(f"DEBUG: Found available models: {valid_models}")

        # 3. Priority Preference List
        preferences = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-001",
            "gemini-pro",
            "gemini-1.0-pro"
        ]
        
        # 4. Find the best match
        for pref in preferences:
            for valid in valid_models:
                if pref in valid:
                    # LangChain needs the name without 'models/' prefix
                    return valid.replace("models/", "")
        
        # 5. If no preference match, take the first valid one
        if valid_models:
            return valid_models[0].replace("models/", "")
            
        # 6. Absolute Fallback (if list fails)
        return "gemini-pro"
        
    except Exception as e:
        print(f"WARNING: Could not list models ({e}). Defaulting to gemini-pro.")
        return "gemini-pro"

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    # Get the model that ACTUALLY exists for this key
    best_model = get_available_model(api_key)
    print(f"DEBUG: Selected Model: {best_model}")

    llm = ChatGoogleGenerativeAI(
        model=best_model, 
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
