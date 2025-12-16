import streamlit as st
import os
from backend.workflow import build_pharma_graph
from backend.rag_engine import rag_system

# Page Config
st.set_page_config(page_title="EY Techathon: Pharma Agent", layout="wide")
st.title("🧬 Pharma Agentic AI: Innovation Engine")
st.caption("Powered by Gemini 1.5, LangGraph, and Real-Time APIs")

# Sidebar
# ... (imports remain the same)

# Sidebar
st.sidebar.header("Setup")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- FIX: Initialize RAG with Key ---
if api_key:
    rag_system.setup(api_key)  # <--- THIS LINE IS CRITICAL

uploaded_file = st.sidebar.file_uploader("Upload Internal Strategy PDF", type=["pdf"])
if uploaded_file and api_key: # Ensure key exists before processing
    # Save and Ingest
    if not os.path.exists("data"): os.makedirs("data")
    path = os.path.join("data", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Ingesting Knowledge Base..."):
        rag_system.ingest(path)
    st.sidebar.success("PDF Indexed!")
# ... (rest of the app)

# Main Chat
query = st.text_area("Enter Strategic Query", 
                     "Can we repurpose Metformin for Anti-Aging indications? Check clinical trials, patent landscape, and market viability.")

if st.button("Generate Strategy") and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    app = build_pharma_graph()
    
    # Run Graph
    with st.status("Orchestrating Agents...", expanded=True) as status:
        st.write("🧠 Master Agent: Analyzing query & generating JSON Plan...")
        
        # We invoke the graph
        inputs = {"user_query": query, "api_key": api_key}
        result = app.invoke(inputs)
        
        # Display Plan
        st.write("📋 **Master Plan Generated:**")
        st.json(result["master_plan"])
        
        # Display Agent Outputs
        st.write("🚜 **Agent Execution Results:**")
        for agent, output in result["agent_outputs"].items():
            with st.expander(f"Agent: {agent}"):
                st.write(output)
        
        status.update(label="Process Complete!", state="complete", expanded=False)

    # Final Report
    st.markdown("---")
    st.header("📄 Strategic Innovation Report")
    st.markdown(result["final_report"])
    
    # Download Button
    st.download_button("Download Report", result["final_report"], file_name="strategy_report.md")

elif not api_key:
    st.warning("Please enter your Gemini API Key to proceed.")
