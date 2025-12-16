import streamlit as st
import os
from backend.workflow import build_pharma_graph
from backend.rag_engine import rag_system

# Page Config
st.set_page_config(page_title="EY Techathon: Pharma Agent", layout="wide")
st.title("🧬 Pharma Agentic AI: Innovation Engine")

# --- 1. Load Secrets & Setup RAG ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key # Set env var for generic tools if needed
    
    # Initialize RAG silently on startup
    rag_system.setup(api_key)
    rag_system.load_directory("data") # Auto-load any PDFs in data/ folder

except FileNotFoundError:
    st.error("🚨 Missing Secrets! Please create .streamlit/secrets.toml with your GOOGLE_API_KEY.")
    st.stop()
except KeyError:
    st.error("🚨 Key Missing! Make sure GOOGLE_API_KEY is defined in secrets.toml.")
    st.stop()

# --- 2. Main Interface ---
# No sidebar clutter. Just the business.

query = st.text_area(
    "Enter Strategic Research Query", 
    "Investigate the feasibility of repurposing Atorvastatin for Alzheimer's disease. Check ongoing trials, patents, and market size.",
    height=100
)

if st.button("🚀 Generate Innovation Strategy"):
    
    app = build_pharma_graph()
    
    with st.status("🤖 Orchestrating AI Agents...", expanded=True) as status:
        st.write("🧠 **Master Agent:** Analyzing query & generating Master Plan...")
        
        # Invoke Graph
        inputs = {"user_query": query, "api_key": api_key}
        result = app.invoke(inputs)
        
        # Display The JSON Plan (Requirement)
        st.write("📋 **Strategic Plan (JSON):**")
        st.json(result["master_plan"])
        
        st.write("⚡ **Executing Worker Agents:**")
        
        # Dynamic tabs for agent outputs
        agent_results = result["agent_outputs"]
        if agent_results:
            tabs = st.tabs(list(agent_results.keys()))
            for i, (agent, output) in enumerate(agent_results.items()):
                with tabs[i]:
                    st.markdown(output)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Final Report Display
    st.divider()
    st.subheader("📄 Strategic Innovation Report")
    st.markdown(result["final_report"])
    
    # Download
    st.download_button(
        "📥 Download Report", 
        result["final_report"], 
        file_name="strategy_report.md"
    )
