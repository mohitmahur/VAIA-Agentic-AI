import streamlit as st
from dotenv import load_dotenv
import os
from agent import MarketAnalystAgent

load_dotenv()

st.set_page_config(page_title="AI Market Analyst", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top left, #1e293b 0%, #0f172a 80%);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .big-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #818cf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }

    .stTextInput>div>div>input {
        border-radius: 12px;
        background-color: rgba(30,41,59,0.8);
        color: #f8fafc;
        border: 1px solid #334155;
        padding: 0.8rem;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #60a5fa;
        box-shadow: 0 0 12px #3b82f6;
    }

    /* Advanced Button Design */
    .stButton>button {
        position: relative;
        overflow: hidden;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        cursor: pointer;
        box-shadow: 0 8px 25px rgba(59,130,246,0.3);
        transition: all 0.25s ease;
        isolation: isolate;
    }

    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0.2), rgba(255,255,255,0));
        transform: skewX(-15deg);
        transition: 0.6s;
        z-index: 1;
    }

    .stButton>button:hover::before {
        left: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.03);
        background: linear-gradient(90deg, #1d4ed8, #4338ca);
        box-shadow: 0 10px 30px rgba(59,130,246,0.4);
    }

    .stButton>button:active {
        transform: scale(0.97);
        box-shadow: 0 4px 12px rgba(59,130,246,0.2);
    }

    /* Results */
    .stMarkdown, .stSubheader {
        color: #f1f5f9;
        animation: fadeIn 0.6s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='big-title'>AI Market Analyst</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Smart Market Insights • Research Summaries • Data Extraction</div>",
    unsafe_allow_html=True,
)

document_file = os.path.join("data", "innovate_q3_2025.txt")
agent = MarketAnalystAgent(document_path=document_file)

query = st.text_input("Enter your query or request:")

col1, col2, col3, col4 = st.columns(4)
auto_route = col1.button("Auto Route")
qa_button = col2.button("General Q&A")
summary_button = col3.button("Summarize")
extract_button = col4.button("Extract Data")

if auto_route and query:
    with st.spinner("Routing to best analytical tool..."):
        tool = agent.route_query(query).tool_name
        if tool == "qa":
            result = agent.perform_general_qa(query)
            st.subheader("Result: General Q&A")
            st.write(result)
        elif tool == "summarize":
            result = agent.get_market_research_findings()
            st.subheader("Result: Summary")
            st.write(result)
        elif tool == "extract":
            result = agent.extract_structured_data()
            st.subheader("Result: Structured Data")
            st.json(result.dict())

elif qa_button and query:
    with st.spinner("Generating analytical insights..."):
        result = agent.perform_general_qa(query)
        st.subheader("Answer")
        st.write(result)

elif summary_button:
    with st.spinner("Summarizing report..."):
        result = agent.get_market_research_findings()
        st.subheader("Market Research Summary")
        st.write(result)

elif extract_button:
    with st.spinner("Extracting structured information..."):
        result = agent.extract_structured_data()
        st.subheader("Structured Competitor Data")
        st.json(result.dict())
