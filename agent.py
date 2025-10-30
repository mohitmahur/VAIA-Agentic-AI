import os
from dotenv import load_dotenv
from typing import List, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from rich.console import Console


def setup_environment():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")


class MarketAnalystAgent:
    def __init__(self, document_path: str):
        self.console = Console()
        self.document_path = document_path

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

        loader = TextLoader(document_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        )

        self.retriever = vectorstore.as_retriever()

    def route_query(self, query: str):

        class RouteQuery(BaseModel):
            tool_name: Literal["qa", "summarize", "extract"] = Field(
                ...,
                description="Best suited tool for the user's query.",
            )

        tool_descriptions = """
        - 'qa': For specific factual questions about the document.
        - 'summarize': For overview-type queries.
        - 'extract': For structured data extraction tasks.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are an expert router. Route the query to one of the following tools:\n{tool_descriptions}",
                ),
                ("human", "Route this query: '{query}'"),
            ]
        )

        routing_chain = prompt | self.llm.with_structured_output(schema=RouteQuery)
        return routing_chain.invoke({"query": query})

    def perform_general_qa(self, question: str):
        self.console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")

        template = """You are a market research analyst AI.
        Use the provided context to answer questions. 
        

        Context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        return answer

    def get_market_research_findings(self):
        summary_question = "Summarize key market research findings: market size, growth, competition, and strategy in bullet points."
        return self.perform_general_qa(summary_question)

    def extract_structured_data(self):
        self.console.print(
            "\n[bold cyan]Task:[/bold cyan] Extracting structured competitor data..."
        )

        class Competitor(BaseModel):
            name: str = Field(description="Competitor company name.")
            market_share: Optional[float] = Field(
                description="Market share percentage."
            )

        class CompetitorAnalysis(BaseModel):
            competitors: List[Competitor]

        with open(self.document_path, "r") as f:
            doc_text = f.read()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract competitor data from the text. Follow the given schema strictly.",
                ),
                ("human", "{text}"),
            ]
        )

        extraction_chain = prompt | self.llm.with_structured_output(
            schema=CompetitorAnalysis
        )

        extracted_data = extraction_chain.invoke({"text": doc_text})
        return extracted_data


if __name__ == "__main__":
    try:
        setup_environment()
        document_file = os.path.join("data", "innovate_q3_2025.txt")

        agent = MarketAnalystAgent(document_path=document_file)
        agent.console.print(
            "[bold magenta] AI Market Analyst Agent Initialized[/bold magenta]"
        )
        agent.console.print(f"Analyzing document: {document_file}")

        agent.console.print("\n" + "=" * 50)
        agent.console.print("[bold magenta]AUTONOMOUS AGENT SESSION[/bold magenta]")
        agent.console.print("=" * 50)
        agent.console.print(
            "\n[bold green]Ask anything about the document. Type 'quit' to exit.[/bold green]"
        )

        while True:
            user_query = input("\nYour request: ")
            if user_query.lower().strip() in ["quit", "exit"]:
                agent.console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            if user_query:
                routed_tool = agent.route_query(user_query)
                agent.console.print(
                    f"[bold dim]Routing to: {routed_tool.tool_name}[/bold dim]"
                )

                if routed_tool.tool_name == "qa":
                    agent.perform_general_qa(user_query)
                elif routed_tool.tool_name == "summarize":
                    agent.get_market_research_findings()
                elif routed_tool.tool_name == "extract":
                    agent.extract_structured_data()

    except Exception as e:
        print(f"Error: {e}")
