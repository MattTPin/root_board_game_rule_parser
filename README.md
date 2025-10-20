
# Root Rules RAG System

This repository implements a **Retrieval-Augmented Generation (RAG) system** for the board game *Root*. The system reads in the rules of the game from PDF documents and stores them in a **Chroma vector database**. You can then query the database, and in combination with a configurable **LangChain LLM**, receive answers to specific questions about the rules of the game!

---

## Setup

This project is designed to run in a Python virtual environment (`venv`). Follow these steps to set it up:

1. **Create a virtual environment**:

```bash
python -m venv venv
```

2. **Activate the virtual environment**:

- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Usage

Two main functions are provided:

### 1. Populate the database

```bash
python -m populate_db
```

This function will:

- Process all PDFs in the `pdfs/` folder.
- Clean the PDFs that are specified in the `PDFS_TO_CLEAN` configuration.
- Read and preprocess the PDFs.
- Break the text into chunks and store them in the **Chroma vector database** for later retrieval.

### 2. Query the database

```bash
python -m query_db "YOUR ROOT QUESTION"
```

This function will:

- Match relevant rules in the **vector database**.
- Query the currently configured **LangChain LLM** to reason over the retrieved chunks.
- Generate a detailed answer to your question about the rules of *Root*.

---

Now you’re ready to explore the rules of *Root* interactively using your RAG system!


## LLM Configuration

This project uses **LangChain** as a universal interface for interacting with multiple Large Language Model (LLM) APIs.

You can dynamically switch between **Anthropic (Claude)**, **OpenAI (GPT)**, and **Mistral** models simply by updating the `.env` configuration.

You can create your own `.env` file by copying the `.env.template` contents and replacing fields with your chosen settings / API keys.

---

### **Supported Providers**

| Provider   | Example Model               |
|------------|----------------------------|
| `anthropic`| `claude-sonnet-4-5-20250929` |
| `openai`   | `gpt-4o-mini`               |
| `mistral`  | `mistral-large-latest`      |

---

### **Selecting a Provider**

Set your preferred provider and model in the `.env` file:

```bash
# Choose which company's API to utilize
# One of: "anthropic", "openai", "mistral"
LLM_PROVIDER=anthropic
```

The system will automatically route requests through the corresponding LangChain wrapper.

---

### **Provider API Credentials**

Each provider requires its own API key and model ID.  
Add these values to your `.env` file (examples below):

```bash
# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_MODEL_ID=claude-sonnet-4-5-20250929

# OpenAI (GPT)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL_ID=gpt-4o-mini

# Mistral
MISTRAL_API_KEY=<REPLACE_ME>
MISTRAL_MODEL_ID=mistral-large-latest
```

> Use a local `.env` file (ignored by `.gitignore`) or a secure secret manager.

---