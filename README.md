# Chat with PDF using Self-Correcting RAG

An intelligent, self-correcting agent designed to answer questions about your PDF documents. This project uses Python, LangChain, LangGraph, and Azure OpenAI to create a robust, stateful chatbot.

## Overview

This repository contains a Python script that implements an advanced Retrieval-Augmented Generation (RAG) system. Unlike basic RAG, this implementation uses `LangGraph` to create a cyclical process where the agent can retrieve information, grade its relevance, and even rewrite the user's question if the initial retrieval is not helpful. This creates a more resilient and accurate question-answering system.

## Key Features

- **📄 PDF Processing**: Loads and splits PDF documents into searchable chunks using `PyPDFLoader`.
- **🧠 In-Memory Vector Store**: Creates a fast, in-memory vector store for efficient semantic search.
- **☁️ Azure OpenAI Integration**: Leverages Azure's powerful models for generating embeddings and chat completions.
- **🔗 Self-Corrective RAG**: Implements a `LangGraph` workflow that assesses the relevance of retrieved context and can decide to rewrite the query or generate an answer.
- **⌨️ Interactive CLI**: Provides a simple and clean command-line interface for chatting with your document.

## Tech Stack

- **Python**
- **LangChain** & **LangGraph** for the core RAG framework and agent state management.
- **Azure OpenAI** for embedding and language models.
- **PyPDF** for PDF document parsing.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- An Azure OpenAI account with a deployed GPT model and an `ada-002` embedding model.
- Git installed on your system.

### 1. Clone the Repository

First, clone the repository to your local machine and navigate into the project directory.
```bash
git clone git@github.com:shohog/selfrag_minion.git
cd selfrag_minion
```

### 2. Set Up a Virtual Environment (Recommended)

It is a best practice to create a virtual environment to manage project-specific dependencies.

- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **On Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies

Install all the required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Configure Your Project

Before running the script, you need to:

1.  **Add Your PDF**: Place the PDF file you want to query into the root directory of the project. The script is configured to look for a file named `constitution_bd.pdf`. If your PDF has a different name, you must update the `PDF_FILE_PATH` variable and the tools description accordingly inside the `chat_with_constitution.py` script.

2.  **Prepare Your API Key**: The script will securely prompt you for your `AZURE_OPENAI_API_KEY` the first time you run it. Make sure you have it ready. All other Azure configurations (like endpoint and deployment names) can be adjusted at the top of the script if needed.

## ▶️ How to Run

With your virtual environment active and the configuration complete, run the main script from the terminal:

```bash
python chat_with_constitution.py
```

The application will first prompt you for your Azure API key. After providing it, the PDF will be processed, and the chat session will begin.

### Example Session
```
$ python chat_with_constitution.py
AZURE_OPENAI_API_KEY: ********************************
Type 'exit' or 'quit' to stop.

You: What is the preamble of the Constitution about?
Assistant: The preamble of the Constitution sets out the fundamental principles and the basic structure of the state, including nationalism, socialism, democracy, and secularism. It reflects the historical context and aspirations of the nation at the commencement of the Constitution on December 16, 1972. The preamble is a guiding statement that cannot be amended.

You: exit
```

---
