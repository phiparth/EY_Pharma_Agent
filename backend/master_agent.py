from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    # TRY/EXCEPT BLOCK FOR ROBUSTNESS
    # We try Flash first, then Pro if Flash fails.
    
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    llm = None
    last_error = None
    
    for model_name in models_to_try:
        try:
            print(f"DEBUG: Attempting to use model: {model_name}")
            test_llm = ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=api_key,
                temperature=0
            )
            # Simple test invocation to check if model exists/works
            test_llm.invoke("test")
            llm = test_llm # If we get here, it works
            break
        except Exception as e:
            print(f"WARNING: Model {model_name} failed: {e}")
            last_error = e
            continue

    if not llm:
        raise ValueError(f"All Gemini models failed. Please check your API Key. Last error: {last_error}")

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
