from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from .master_agent import generate_master_plan
from .worker_agents import AGENT_MAP
from .rag_engine import rag_system
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

class GraphState(TypedDict):
    user_query: str
    master_plan: dict
    agent_outputs: dict
    final_report: str
    api_key: str

def plan_step(state: GraphState):
    """Step 1: Generate the Master Plan"""
    print("--- ORCHESTRATOR: Generating Plan ---")
    plan = generate_master_plan(state["user_query"], state["api_key"])
    return {"master_plan": plan.dict()}

def execute_step(state: GraphState):
    """Step 2: Execute all agents in the plan"""
    print("--- ORCHESTRATOR: Executing Agents ---")
    plan = state["master_plan"]
    results = {}
    
    for task in plan["required_agents"]:
        agent_name = task["agent_name"]
        instruction = task["specific_instruction"]
        
        # Execute specific logic based on agent type
        if agent_name == "InternalKnowledgeAgent":
            output = rag_system.query(instruction)
        elif agent_name in AGENT_MAP:
            # Pass specific context args if needed by the tool signature
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
                # Web Intelligence
                output = AGENT_MAP[agent_name].invoke(instruction)
        else:
            output = "Error: Agent not found."
            
        results[agent_name] = output
        
    return {"agent_outputs": results}

def synthesize_step(state: GraphState):
    """Step 3: Synthesize Final Report"""
    print("--- ORCHESTRATOR: Synthesizing Report ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=state["api_key"])
    
    summary_prompt = f"""
    You are a Pharmaceutical Strategy Consultant.
    
    Context:
    The user asked: "{state['user_query']}"
    
    We have gathered the following data from our agents:
    {json.dumps(state['agent_outputs'], indent=2)}
    
    Task:
    Write a "Strategic Innovation Story" report (Markdown format).
    Structure:
    1. **Executive Summary**: High-level feasibility.
    2. **Clinical Landscape**: Summary of trials and competitors.
    3. **IP & Legal**: Patent risks (FTO).
    4. **Commercial Viability**: Market size and trends.
    5. **Recommendation**: Go/No-Go decision and next steps.
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
