import streamlit as st
import os

st.set_page_config(page_title="Pharma Agent", layout="wide")
st.title("🧬 Pharma Agentic AI (Powered by OpenAI)")

# --- OPENAI KEY HANDLING ---
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("🚨 Missing OPENAI_API_KEY in Secrets.")
    st.stop()

# Import Backend
from backend.workflow import build_pharma_graph
from backend.rag_engine import rag_system

# Initialize RAG
try:
    rag_system.setup(api_key)
    if hasattr(rag_system, 'load_directory'): rag_system.load_directory("data")
except: pass

query = st.text_area("Research Query", "Feasibility of Metformin for Anti-Aging.")

if st.button("🚀 Run Analysis"):
    with st.status("🤖 Orchestrating Agents...", expanded=True):
        try:
            app = build_pharma_graph()
            # Run Graph
            result = app.invoke({"user_query": query, "api_key": api_key})
            
            st.write("📋 **Strategic Plan:**")
            st.json(result["master_plan"])
            
            st.write("⚡ **Agent Insights:**")
            for agent, output in result["agent_outputs"].items():
                with st.expander(agent): st.markdown(output)
            
            st.subheader("📄 Final Strategy Report")
            st.markdown(result["final_report"])
            
            st.download_button("Download Report", result["final_report"], file_name="report.md")
            
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
