# Chat with PDF using LangChain and Azure OpenAI

This project allows you to chat with a PDF document using a Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and Azure OpenAI services.

## Features

- **PDF Processing**: Loads and splits a PDF document into manageable chunks.
- **Vector Store**: Creates an in-memory vector store for efficient information retrieval.
- **Azure OpenAI Integration**: Utilizes Azure's powerful models for embeddings and chat responses.
- **Self-Corrective RAG**: Implements a graph-based workflow that can grade the relevance of retrieved documents and rewrite questions if needed.
- **Command-Line Interface**: Provides an interactive chat loop to ask questions about the PDF.

## Prerequisites

- Python 3.8 or higher
- An Azure OpenAI account with access to GPT and embedding models.
- Git installed on your system.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:shohog/selfrag_minion.git
cd selfrag_minion
python chat_with_constitution.py
