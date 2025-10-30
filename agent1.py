import os
from dotenv import load_dotenv
from typing import List, Optional

# LangChain components
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_extraction_chain

# Pydantic for structured data extraction
from pydantic import BaseModel, Field

# Rich for pretty printing
from rich.console import Console


def setup_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")


class MarketAnalystAgent:
    """
    An AI agent that analyzes a market research document.
    """

    def __init__(self, document_path: str):
        """
        Initializes the agent by setting up the RAG pipeline.
        
        Technical Justification:
        - We use a RAG (Retrieval-Augmented Generation) pipeline to ensure the LLM's answers
          are grounded in the provided document, reducing hallucinations and improving accuracy.
        - TextLoader: Simple and effective for .txt files.
        - RecursiveCharacterTextSplitter: Chosen because it's robust for generic text. It tries
          to split on semantic boundaries (paragraphs, sentences) before falling back to characters,
          which helps keep related context together in chunks.        
        - GoogleGenerativeAIEmbeddings: A high-performance model for converting text to vector embeddings using Gemini.
        - FAISS: A highly efficient, in-memory vector store for fast similarity searches. It's
          perfect for single-document analysis without the overhead of a full database.
        """
        self.document_path = document_path # Store the document path as an instance attribute
        self.console = Console()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True)
        
        # 1. Load the document
        loader = TextLoader(document_path)
        docs = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Create vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
        
        # 4. Create a retriever
        self.retriever = vectorstore.as_retriever()

    def perform_general_qa(self, question: str):
        """
        Performs general Question & Answering based on the document.
        """
        self.console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
        
        # Technical Justification for the chain:
        # - A prompt template structures the input for the LLM, clearly separating the
        #   retrieved context from the user's question.
        # - `RunnablePassthrough` is used to pass the original question through the chain
        #   alongside the retrieved context.
        # - `StrOutputParser` ensures the LLM's output is a clean string.
        template = """Answer the question based only on the following context:
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
        self.console.print(f"[bold green]Answer:[/bold green] {answer}")

    def get_market_research_findings(self):
        """
        Generates a summary of the key market research findings.
        This is a specific form of Q&A, asking for a summary.
        """
        summary_question = "Summarize the key market research findings from the document. Include market size, growth, competitive landscape, and strategic priorities."
        self.perform_general_qa(summary_question)

    def extract_structured_data(self):
        """
        Extracts structured data (competitor analysis) from the document.
        """
        self.console.print("\n[bold cyan]Task:[/bold cyan] Extracting Structured Competitor Data...")

        # Technical Justification for structured extraction:
        # - Pydantic Models: We define a desired JSON schema using Pydantic. This provides a clear,
        #   type-safe structure for the data we want to extract.
        # - `create_extraction_chain`: This LangChain function is optimized for extraction. It uses
        #   the LLM's function-calling capabilities to reliably populate the Pydantic model,
        #   turning unstructured text into structured data. This is more robust than just
        #   prompting for JSON.
        class Competitor(BaseModel):
            name: str = Field(description="The name of the competitor company.")
            market_share: Optional[float] = Field(description="The market share percentage of the competitor, as a float.")

        class CompetitorAnalysis(BaseModel):
            """A list of competitors and their market share."""
            competitors: List[Competitor]

        # We pass the entire document content for extraction to ensure all data is captured.
        with open(self.document_path, 'r') as f: # Access the stored document path
            doc_text = f.read()

        # Modern approach using with_structured_output
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are an expert extraction agent. Extract relevant information from the provided text and format it according to the schema. Only extract information from the text."),
             ("human", "{text}")]
        )
        
        extraction_chain = prompt | self.llm.with_structured_output(schema=CompetitorAnalysis)
        
        extracted_data = extraction_chain.invoke({"text": doc_text})
        
        self.console.print("[bold green]Extracted Data:[/bold green]")
        self.console.print(extracted_data)


if __name__ == "__main__":
    try:
        # --- Setup ---
        setup_environment()
        
        # --- Initialization ---
        # This creates the agent and prepares the document for analysis (loads, chunks, embeds).
        document_file = os.path.join("data", "innovate_q3_2025.txt")
        agent = MarketAnalystAgent(document_path=document_file)
        agent.console.print("[bold magenta]AI Market Analyst Agent Initialized.[/bold magenta]")
        agent.console.print("Ready to perform tasks on 'innovate_q3_2025.txt'.")

        # --- Task 1: General Q&A ---
        agent.console.print("\n" + "="*50)
        agent.console.print("[bold magenta]TASK 1: GENERAL Q&A[/bold magenta]")
        agent.console.print("="*50)
        agent.perform_general_qa("What is Innovate Inc.'s market share?")
        agent.perform_general_qa("What are the main threats to Innovate Inc.?")

        # --- Task 2: Market Research Findings ---
        agent.console.print("\n" + "="*50)
        agent.console.print("[bold magenta]TASK 2: MARKET RESEARCH FINDINGS[/bold magenta]")
        agent.console.print("="*50)
        agent.get_market_research_findings()

        # --- Task 3: Structured Data Extraction ---
        agent.console.print("\n" + "="*50)
        agent.console.print("[bold magenta]TASK 3: STRUCTURED DATA EXTRACTION[/bold magenta]")
        agent.console.print("="*50)
        agent.extract_structured_data()

        # --- Task 4: Interactive Q&A Session ---
        agent.console.print("\n" + "="*50)
        agent.console.print("[bold magenta]INTERACTIVE Q&A SESSION[/bold magenta]")
        agent.console.print("="*50)

        start_interactive = input("Do you want to ask more questions? (y/n): ").lower().strip()

        if start_interactive == 'y':
            agent.console.print("\n[bold green]Great! You can now ask questions. Type 'quit' to exit.[/bold green]")
            while True:
                user_question = input("\nYour question: ")
                if user_question.lower().strip() in ['quit', 'exit']:
                    agent.console.print("[bold yellow]Exiting interactive session. Goodbye![/bold yellow]")
                    break
                if user_question:
                    agent.perform_general_qa(user_question)
        else:
            agent.console.print("\n[bold yellow]Skipping interactive session. Goodbye![/bold yellow]")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")