from langgraph.graph import StateGraph, END
from typing import TypedDict
from .master_agent import generate_master_plan
from .worker_agents import AGENT_MAP
from .rag_engine import rag_system
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json

class GraphState(TypedDict):
    user_query: str
    master_plan: dict
    agent_outputs: dict
    final_report: str
    api_key: str

def plan_step(state: GraphState):
    print("--- ORCHESTRATOR: Generating Plan ---")
    plan = generate_master_plan(state["user_query"], state["api_key"])
    return {"master_plan": plan.dict()}

def execute_step(state: GraphState):
    print("--- ORCHESTRATOR: Executing Agents ---")
    plan = state["master_plan"]
    results = {}
    
    for task in plan["required_agents"]:
        agent_name = task["agent_name"]
        instruction = task["specific_instruction"]
        
        if agent_name == "InternalKnowledgeAgent":
            output = rag_system.query(instruction)
        elif agent_name in AGENT_MAP:
            # Context injection for specific tools
            if agent_name == "ClinicalTrialsAgent":
                output = AGENT_MAP[agent_name].invoke({
                    "instruction": instruction,
                    "molecule": plan["molecule"], 
                    "indication": plan["indication"]
                })
            elif agent_name == "PatentLandscapeAgent":
                 output = AGENT_MAP[agent_name].invoke({
                    "instruction": instruction,
                    "molecule": plan["molecule"]
                })
            elif agent_name == "IQVIAInsightsAgent":
                 output = AGENT_MAP[agent_name].invoke({
                    "instruction": instruction,
                    "therapeutic_area": plan["therapeutic_area"]
                })
            elif agent_name == "EXIMTrendsAgent":
                 output = AGENT_MAP[agent_name].invoke({
                    "instruction": instruction,
                    "molecule": plan["molecule"]
                })
            else:
                output = AGENT_MAP[agent_name].invoke(instruction)
        else:
            output = "Error: Agent not found."
            
        results[agent_name] = output
        
    return {"agent_outputs": results}

def synthesize_step(state: GraphState):
    print("--- ORCHESTRATOR: Synthesizing Report ---")
    
    # Fallback Logic for Synthesis as well
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    llm = None
    for model in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=state["api_key"])
            llm.invoke("test")
            break
        except:
            continue
            
    if not llm:
        return {"final_report": "Error: Could not connect to any Gemini model for reporting."}
    
    summary_prompt = f"""
    You are a Pharmaceutical Strategy Consultant.
    User Query: "{state['user_query']}"
    
    Data Gathered:
    {json.dumps(state['agent_outputs'], indent=2)}
    
    Write a Strategic Innovation Story report (Markdown).
    Include: Executive Summary, Clinical Landscape, IP Risks, Commercial Viability, Recommendation.
    """
    
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"final_report": response.content}

def build_pharma_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("Planner", plan_step)
    workflow.add_node("Executor", execute_step)
    workflow.add_node("Synthesizer", synthesize_step)
    workflow.set_entry_point("Planner")
    workflow.add_edge("Planner", "Executor")
    workflow.add_edge("Executor", "Synthesizer")
    workflow.add_edge("Synthesizer", END)
    return workflow.compile()
