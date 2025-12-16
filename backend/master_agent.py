import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from .models import MasterPlan
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def get_working_model_name(api_key: str):
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prefer Flash -> Pro -> Fallback
        for pref in ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]:
            if pref in models: return pref.replace("models/", "")
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

def generate_master_plan(user_query: str, api_key: str) -> MasterPlan:
    model_name = get_working_model_name(api_key)
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    
    parser = PydanticOutputParser(pydantic_object=MasterPlan)
    prompt = PromptTemplate(
        template="Analyze query: {query}\nGenerate MasterPlan JSON.\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return (prompt | llm | parser).invoke({"query": user_query})
