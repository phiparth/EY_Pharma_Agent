import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st

def get_working_model_name(api_key: str):
    """
    Connects to Google API, lists all available models, and finds the first one 
    that supports text generation and actually works.
    """
    genai.configure(api_key=api_key)
    try:
        # 1. Get all models
        models = list(genai.list_models())
        
        # 2. Filter for text generation models
        text_models = [
            m.name for m in models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # 3. Priority List (Newest first)
        priority_order = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-pro-001",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        selected_model = None
        
        # Try finding a match in our priority list
        for priority in priority_order:
            if priority in text_models:
                selected_model = priority
                break
        
        # If no priority match, take the first valid one we found
        if not selected_model and text_models:
            selected_model = text_models[0]
            
        if not selected_model:
            # Fallback hard if list is empty (rare)
            selected_model = "gemini-1.5-flash"

        print(f"DEBUG: Auto-Selected Model: {selected_model}")
        return selected_model.replace("models/", "") # LangChain usually prefers just the name

    except Exception as e:
        print(f"Error listing models: {e}")
        # Absolute backup
        return "gemini-1.5-flash" 

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    
    # DYNAMICALLY FIND THE CORRECT MODEL
    best_model = get_working_model_name(api_key)
    
    # Configure LLM with the found model
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
