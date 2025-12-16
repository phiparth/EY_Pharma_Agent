import streamlit as st
import os

st.set_page_config(page_title="Pharma Agent", layout="wide")
st.title("🧬 Pharma Agentic AI")

# --- API KEY HANDLING ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 Secrets missing! Add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

from backend.workflow import build_pharma_graph
from backend.rag_engine import rag_system

# Initialize RAG (Silent Fail if no file)
try:
    rag_system.setup(api_key)
    if hasattr(rag_system, 'load_directory'): rag_system.load_directory("data")
except: pass

query = st.text_area("Research Query", "Feasibility of Metformin for Anti-Aging.")

if st.button("🚀 Run Analysis"):
    with st.status("🤖 Agents Working...", expanded=True):
        try:
            app = build_pharma_graph()
            # Explicitly passing API key in inputs
            result = app.invoke({"user_query": query, "api_key": api_key})
            
            st.write("📋 **Plan:**")
            st.json(result["master_plan"])
            
            st.write("⚡ **Results:**")
            for agent, output in result["agent_outputs"].items():
                with st.expander(agent): st.markdown(output)
            
            st.subheader("📄 Report")
            st.markdown(result["final_report"])
            
            st.download_button("Download Report", result["final_report"], file_name="report.md")
        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Tip: If 404 Model Not Found, check if your API Key supports Gemini 1.5 Flash.")
