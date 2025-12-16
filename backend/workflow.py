from langgraph.graph import StateGraph, END
from typing import TypedDict
from .master_agent import generate_master_plan, get_fallback_llm
from .worker_agents import AGENT_MAP
from .rag_engine import rag_system
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
            # Inject context
            context = {}
            if agent_name == "ClinicalTrialsAgent":
                context = {"molecule": plan["molecule"], "indication": plan["indication"]}
            elif agent_name in ["PatentLandscapeAgent", "EXIMTrendsAgent"]:
                context = {"molecule": plan["molecule"]}
            elif agent_name == "IQVIAInsightsAgent":
                context = {"therapeutic_area": plan["therapeutic_area"]}
            
            # Combine instruction with context if needed, or pass as dict
            # For simplicity in this robust version, we try passing dict if tool supports it, else string
            try:
                if context:
                    # Merge instruction into context
                    context["instruction"] = instruction
                    output = AGENT_MAP[agent_name].invoke(context)
                else:
                    output = AGENT_MAP[agent_name].invoke(instruction)
            except Exception as e:
                output = f"Tool Error: {str(e)}"
        else:
            output = "Error: Agent not found."
            
        results[agent_name] = output
        
    return {"agent_outputs": results}

def synthesize_step(state: GraphState):
    print("--- ORCHESTRATOR: Synthesizing Report ---")
    
    # Use the same fallback logic
    llm = get_fallback_llm(state["api_key"])
    
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
