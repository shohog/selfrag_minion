import os
import getpass
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

# --- Environment Setup ---
def set_env(key: str, default: str | None = None, secret: bool = False) -> None:
    """
    Set an environment variable if it does not already exist.
    """
    if key in os.environ:
        return

    if default is not None:
        os.environ[key] = default
    elif secret:
        os.environ[key] = getpass.getpass(f"{key}: ")
    else:
        raise ValueError(f"{key} must be set")

def setup_environment():
    """Set all necessary environment variables."""
    set_env(
        "AZURE_OPENAI_ENDPOINT",
        "https://assessment-test-3-found-resource.cognitiveservices.azure.com/",
    )
    set_env("AZURE_OPENAI_API_KEY", secret=True)
    set_env("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    set_env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    set_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

# --- Model Initialization ---
def initialize_models():
    """Initialize and return the language models."""
    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version="2023-05-15",
    )
    response_model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0,
    )
    grader_model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0,
    )
    return embedding_model, response_model, grader_model

# --- PDF Processing and Retrieval ---
def create_retriever(embedding_model):
    """Load, split, and create a retriever for the PDF."""
    PDF_FILE_PATH = "constitution_bd.pdf"
    loader = PyPDFLoader(PDF_FILE_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embedding_model
    )
    return vectorstore.as_retriever()

# Global retriever for the tool
retriever = None

@tool
def retrieve_from_resume(query: str) -> str:
    """Search and return the most relevant information from the Constitution of Bangladesh using a retriever."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# --- LangGraph Nodes ---
def generate_query_or_respond(state: MessagesState, response_model):
    """Decide to retrieve or respond directly."""
    response = response_model.bind_tools([retrieve_from_resume]).invoke(state["messages"])
    return {"messages": [response]}

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")

def grade_documents(state: MessagesState, grader_model) -> Literal["generate_answer", "rewrite_question"]:
    """Determine if retrieved documents are relevant."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState, response_model):
    """Rewrite the user's original question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState, response_model):
    """Generate a final answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# --- Main Application ---
def main():
    """Main function to run the chat application."""
    global retriever
    setup_environment()
    embedding_model, response_model, grader_model = initialize_models()
    retriever = create_retriever(embedding_model)

    workflow = StateGraph(MessagesState)

    workflow.add_node("generate_query_or_respond", lambda state: generate_query_or_respond(state, response_model))
    workflow.add_node("retrieve", ToolNode([retrieve_from_resume]))
    workflow.add_node("rewrite_question", lambda state: rewrite_question(state, response_model))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, response_model))

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: grade_documents(state, grader_model)
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    graph = workflow.compile()

    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        
        for chunk in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for _, update in chunk.items():
                if "messages" in update and update["messages"]:
                    print("Assistant:", update["messages"][-1].content)
                    print()

if __name__ == "__main__":
    main()
