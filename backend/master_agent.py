import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st

def get_working_model_name(api_key: str):
    """
    Connects to Google API to find which models are actually available 
    for this specific API Key.
    """
    genai.configure(api_key=api_key)
    try:
        # List all models available to this key
        models = list(genai.list_models())
        
        # Filter for models that support text generation
        valid_models = [
            m.name for m in models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Priority list: We prefer 1.5 Flash/Pro, then standard Pro
        preferences = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        # 1. Check if any preferred alias exists exactly in the valid list
        for pref in preferences:
            # The API returns names like 'models/gemini-pro', so we check for ends_with
            match = next((m for m in valid_models if m.endswith(pref)), None)
            if match:
                # Remove 'models/' prefix if LangChain adds it automatically, 
                # but usually passing the full name 'models/gemini-...' is safer now.
                return match.replace("models/", "")

        # 2. If no preferred match, just take the first valid Gemini model
        gemini_fallback = next((m for m in valid_models if "gemini" in m), None)
        if gemini_fallback:
            return gemini_fallback.replace("models/", "")

        return "gemini-pro" # Absolute fallback

    except Exception as e:
        print(f"Error listing models: {e}")
        return "gemini-pro" # Blind fallback

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    
    # DYNAMICALLY FIND THE CORRECT MODEL
    best_model = get_working_model_name(api_key)
    print(f"DEBUG: Selected Model: {best_model}")

    try:
        llm = ChatGoogleGenerativeAI(
            model=best_model, 
            google_api_key=api_key,
            temperature=0
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize model {best_model}: {e}")

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
