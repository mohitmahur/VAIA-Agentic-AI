# AI Market Analyst Agent

The **AI Market Analyst** is a Streamlit-based intelligent agent powered by **Google Gemini (via LangChain)** that can perform:
- Market Q&A  
- Executive-level Summarization  
- Structured Competitor Data Extraction  

It reads and analyzes market research text file (`innovate_q3_2025.txt`) and provides deep, AI-powered insights.

---

## Project Structure

```
VIAI/
│
├── data/
│   ├── innovate_q3_2025.txt      # Market research data file
│                   
│
├── .env                          # Contains GOOGLE_API_KEY
├── agent.py                      # Core AI agent logic (LangChain + Gemini)
├── app.py                        # Streamlit frontend app
└── Readme.md                     # Project documentation
├── requirements.txt              # Python dependencies 
                
```

---

## Setup Instructions

### Step 1-  Clone the Repository
```bash
git clone https://github.com/mohitmahur/VAIA-Agentic-AI
cd VAIA-Agentic-AI
```



### Step 2- Install Dependencies
```bash
pip install -r requirements.txt
```

### Step $- Add Your Google API Key
Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

> You can generate this key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

---

##  Run the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your browser.

---
##  Design Decisions (Crucial)
| **Design Aspect**                          | **Chosen Approach**                                              | **Why It’s Better (vs Alternatives)**                                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Chunking Strategy**                      | **Adaptive RecursiveCharacterTextSplitter (600–1200 / 100–200)** | Dynamically adjusts based on doc size → balances coherence and speed; recursive fallback preserves sentence boundaries  |
| **Chunking vs Sentence / Token Splitters** | RecursiveCharacter > TokenTextSplitter / SentenceTextSplitter    | Sentence-based can over-fragment; token-based ignores semantic flow; recursive keeps natural structure                  |
| **Embedding Model**                        | `text-embedding-004` (Google Generative AI)                      | Outperforms OpenAI `text-embedding-3-small` in multilingual and factual domains; 90%+ correlation with Gemini reasoning |
| **Vector DB**                              | FAISS                                                            | Local, fast, memory-efficient; better for research/intern-level prototypes than Pinecone or Qdrant which require setup  |
| **LLM**                                    | `gemini-2.5-flash`                                               | Better reasoning speed than Claude 3 Haiku / GPT-4o-mini; ideal for document retrieval tasks; cheaper + fast response   |
| **Adaptivity**                             | Auto-tunes chunk size based on doc length                        | system scales automatically                                                                    |
| **Practicality**                           | Print logs via Rich Console                                      | Helpful for debugging & recruiter evaluation; makes system transparent                                             |



| **Model**                      | **Speed**    | **Context Handling** | **Cost-Efficiency** | **Reasoning Coherence** | **Suitability**             |
| ------------------------------ | ------------ | -------------------- | ------------------- | ----------------------- | --------------------------- |
| **Gemini-2.5-Flash**           | Fast       | Excellent          | Very Low         | Strong                | Real-time doc QA            |
| **GPT-4o-mini (OpenAI)**       | Fast       | Strong             | Moderate         | Excellent             | General-purpose             |
| **Claude 3 Haiku (Anthropic)** | Very Fast | Limited           | Low              | Simplistic responses | Short summaries             |
| **Mistral Large**              | Slower    | High               | High             | Good logic            | Large-scale reports         |
| **Llama 3.1 (Meta)**           | Moderate  | oken limits      | Free (open)      | Weak coherence       | Offline / open source setup |


Conclusion:
Gemini-2.5-Flash + text-embedding-004 + Adaptive Recursive Chunking gives the highest trade-off score among all —
balancing speed, context precision, semantic accuracy, and compute cost — ideal for real-world document reasoning agents.


### 4. **Data Extraction Prompt

The extraction prompt uses strict Pydantic schema enforcement to ensure well-structured JSON outputs.
By defining CompetitorAnalysis and Competitor models, Gemini is guided to adhere exactly to schema constraints, reducing hallucination and formatting errors.


### **API Usage (Gemini Models)

LLM Model: gemini-2.5-flash

Embedding Model: models/text-embedding-004
These are part of the Google Generative AI suite, accessed via LangChain’s ChatGoogleGenerativeAI and GoogleGenerativeAIEmbeddings.

### 5. **API Usage Examples**

#### 🔹 General Q&A
```python
response = agent.perform_general_qa("What are the top competitors in the Q3 report?")
")
print(response)
```

#### 🔹 Summarization
```python
summary = agent.get_market_research_findings()
print(summary)
```

#### 🔹 Structured Data Extraction
```python
data = agent.extract_structured_data()
print(data.dict())
```

#### 🔹 Auto Routing
```python
route = agent.route_query("Summarize the competitor section").tool_name
print(route)
```

---

## Core Features

- **Auto Tool Routing** based on natural language intent detection  
- **Market Q&A** using contextual embeddings  
- **Comprehensive Summarization** of full documents  
- **Structured JSON Data Extraction** for downstream analytics  
- **Professional Streamlit Interface** with animated modern UI elements  

---

## Tech Stack

### Core Tech Stack

#### LLM: Google Gemini 2.5 Flash
#### Embeddings: Google Text Embedding 004
#### Frameworks: LangChain, FAISS, Streamlit
#### Language: Python 3.10+


### **Features

## Automated query routing (QA / Summarize / Extract)
## Context-aware document retrieval via FAISS
## Gemini-powered structured output generation
## Professional Streamlit UI for business analysts
---


## License

This project is for **educational and research** use only.  
All rights reserved © 2025.

---

**Author:** Mohit Mahur 

