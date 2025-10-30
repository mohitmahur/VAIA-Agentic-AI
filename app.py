import streamlit as st
from dotenv import load_dotenv
import os

from agent import MarketAnalystAgent  # ← change to your file name!

load_dotenv()

document_file = os.path.join("data", "innovate_q3_2025.txt")
agent = MarketAnalystAgent(document_path=document_file)

st.set_page_config(page_title="AI Market Analyst", layout="centered")

st.title("📊 AI Market Analyst Agent")
st.write("Perform Market Q&A • Summarization • Structured Data Extraction")

# --- User Input ---
query = st.text_input("Ask a question or describe your request:")

# --- Buttons Row ---
col1, col2, col3, col4 = st.columns(4)

auto_route = col1.button("🤖 Auto Route")
qa_button = col2.button("💬 General Q&A")
summary_button = col3.button("📚 Summarize Findings")
extract_button = col4.button("🧱 Extract Structured Data")

# --- Logic Execution ---
if auto_route and query:
    with st.spinner("🔍 Routing to best tool..."):
        tool = agent.route_query(query).tool_name

        if tool == "qa":
            result = agent.perform_general_qa(query)
            st.success("✅ Routed: General Q&A")
            st.write(result)

        elif tool == "summarize":
            result = agent.get_market_research_findings()
            st.success("✅ Routed: Summarization")
            st.write(result)

        elif tool == "extract":
            result = agent.extract_structured_data()
            st.success("✅ Routed: Structured Extraction")
            st.json(result.dict())


elif qa_button and query:
    with st.spinner("💬 Generating answer..."):
        result = agent.perform_general_qa(query)
        st.markdown("### 💬 Answer")
        st.write(result)


elif summary_button:
    with st.spinner("📚 Summarizing document..."):
        result = agent.get_market_research_findings()
        st.markdown("### 📚 Market Research Summary")
        st.write(result)


elif extract_button:
    with st.spinner("🧱 Extracting structured competitor data..."):
        result = agent.extract_structured_data()
        st.markdown("### 🧱 Extracted Competitor Data")
        st.json(result.dict())
