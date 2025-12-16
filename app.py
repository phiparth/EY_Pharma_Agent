import streamlit as st
import os
from backend.workflow import build_pharma_graph
from backend.rag_engine import rag_system

# Page Config
st.set_page_config(page_title="EY Techathon: Pharma Agent", layout="wide")
st.title("🧬 Pharma Agentic AI: Innovation Engine")

# --- 1. Load Secrets & Setup RAG ---
try:
    # Check if secrets exist
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key 
        
        # Initialize RAG
        rag_system.setup(api_key)
        
        # Load Directory (Robust Call)
        if hasattr(rag_system, 'load_directory'):
            rag_system.load_directory("data")
        else:
            st.error("Error: backend/rag_engine.py is outdated. Please update the file.")
            
    else:
        st.error("🚨 Missing 'GOOGLE_API_KEY' in .streamlit/secrets.toml")
        st.stop()

except FileNotFoundError:
    st.error("🚨 Missing Secrets File! Create .streamlit/secrets.toml")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during setup: {e}")
    st.stop()

# --- 2. Main Interface ---
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
        try:
            inputs = {"user_query": query, "api_key": api_key}
            result = app.invoke(inputs)
            
            # Display Plan
            st.write("📋 **Strategic Plan (JSON):**")
            st.json(result["master_plan"])
            
            # Display Agent Outputs
            st.write("⚡ **Executing Worker Agents:**")
            agent_results = result["agent_outputs"]
            
            if agent_results:
                tabs = st.tabs(list(agent_results.keys()))
                for i, (agent, output) in enumerate(agent_results.items()):
                    with tabs[i]:
                        st.markdown(output)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

            # Final Report
            st.divider()
            st.subheader("📄 Strategic Innovation Report")
            st.markdown(result["final_report"])
            
            # Download
            st.download_button(
                "📥 Download Report", 
                result["final_report"], 
                file_name="strategy_report.md"
            )
            
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
