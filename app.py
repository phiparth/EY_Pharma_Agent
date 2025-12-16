import streamlit as st
import os

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="EY Techathon: Pharma Agent", layout="wide")
st.title("🧬 Pharma Agentic AI: Innovation Engine")

# --- CRITICAL: BRIDGE SECRETS TO ENVIRONMENT ---
# This block moves the key from Streamlit's secret vault into the OS environment
# where LangChain expects to find it.

api_key = None

try:
    # Check if the key exists in Streamlit Secrets
    if "GOOGLE_API_KEY" in st.secrets:
        secret_value = st.secrets["GOOGLE_API_KEY"]
        
        # Verify it's not empty
        if secret_value:
            os.environ["GOOGLE_API_KEY"] = secret_value
            api_key = secret_value
        else:
            st.error("🚨 Secret 'GOOGLE_API_KEY' is empty in Streamlit Settings.")
            st.stop()
    else:
        # Fallback for local development if not using secrets.toml but using .env
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        st.error("🚨 No API Key found! Please add GOOGLE_API_KEY to Streamlit Secrets.")
        st.info("Go to: Manage App > Settings > Secrets > Paste: GOOGLE_API_KEY = 'AIza...'")
        st.stop()

except Exception as e:
    st.error(f"🚨 Error loading secrets: {e}")
    st.stop()

# --- IMPORT BACKEND (After setting Env Vars) ---
# We import these HERE, otherwise they might load before the API key is set
from backend.workflow import build_pharma_graph
from backend.rag_engine import rag_system

# --- INITIALIZE RAG ---
# We pass the key explicitly just to be safe
try:
    rag_system.setup(api_key)
    # Compatibility check for the method
    if hasattr(rag_system, 'load_directory'):
        rag_system.load_directory("data")
except Exception as e:
    # Non-fatal error (app can run without PDF RAG)
    print(f"RAG Warning: {e}")

# --- MAIN UI ---
query = st.text_area(
    "Enter Strategic Research Query", 
    "Investigate the feasibility of repurposing Atorvastatin for Alzheimer's disease. Check ongoing trials, patents, and market size.",
    height=100
)

if st.button("🚀 Generate Innovation Strategy"):
    
    # Pass the key into the graph builder if needed, or rely on the env var we just set
    app = build_pharma_graph()
    
    with st.status("🤖 Orchestrating AI Agents...", expanded=True) as status:
        st.write("🧠 **Master Agent:** Analyzing query & generating Master Plan...")
        
        try:
            # Explicitly pass api_key in the state
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
            st.write("Troubleshooting Tip: If this is a '404 Model Not Found', the API Key works but the model name is unavailable in your region.")
