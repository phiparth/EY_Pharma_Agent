from langchain_openai import ChatOpenAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

    parser = PydanticOutputParser(pydantic_object=MasterPlan)

    # UPDATED PROMPT: Explicitly lists EXIM Trends Agent capabilities 
    system_prompt = """
    You are the *Master Orchestrator Agent* for a pharmaceutical innovation engine. 
    Analyze the user query and generate a JSON plan.
    
    ## Available Worker Agents:
    1. *ClinicalTrialsAgent*: Checks active studies and pipelines.
    2. *IQVIAInsightsAgent*: Market size, CAGR, and competitor sales data.
    3. *PatentLandscapeAgent*: Patent expiry, IP risks, and FTO.
    4. *EXIMTrendsAgent*: **CRITICAL**: Supply chain, API import/export volumes, and sourcing risks.
    5. *WebIntelligenceAgent*: Scientific guidelines, news, and RWE.
    6. *InternalKnowledgeAgent*: Internal PDF strategy documents.

    ## Task Instructions:
    1. Extract molecule, indication, and therapeutic area.
    2. Select **all relevant agents**. (Always include EXIMTrendsAgent if feasibility or supply chain is relevant).
    3. Generate specific instructions for each agent.

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
