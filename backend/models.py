from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class AgentTask(BaseModel):
    agent_name: Literal[
        "ClinicalTrialsAgent", 
        "IQVIAInsightsAgent", 
        "PatentLandscapeAgent", 
        "EXIMTrendsAgent", 
        "WebIntelligenceAgent", 
        "InternalKnowledgeAgent"
    ]
    specific_instruction: str = Field(..., description="Precise instruction for the worker agent.")

class MasterPlan(BaseModel):
    molecule: str = Field(..., description="Target molecule name")
    indication: str = Field(..., description="Target disease or condition")
    therapeutic_area: str = Field(..., description="General therapeutic area")
    required_agents: List[AgentTask] = Field(..., description="List of agents to activate")
    research_goal: str = Field(..., description="Summary of the overall objective")
