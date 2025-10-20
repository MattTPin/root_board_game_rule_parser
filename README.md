
# Root Rules RAG System

This repository implements a **Retrieval-Augmented Generation (RAG) system** for the board game *Root*. The system reads in the rules of the game from PDF documents and stores them in a **Chroma vector database**. You can then query the database, and in combination with a configurable **LangChain LLM**, receive answers to specific questions about the rules of the game!

---

## Examples

### Example 1:
Input
```bash
python -m query_db "How does battling work in Root?"
```

Ouput
```bash
When battling, the attacker chooses a clearing where they have warriors and selects another faction with pieces in that clearing as the defender. The battle occurs in two steps:

1. Roll Dice:
- Roll two dice
- Attacker deals hits equal to the higher roll
- Defender deals hits equal to the lower roll
- Each player can roll no more hits than their number of warriors in the clearing

2. Remove Pieces:
- Both players remove pieces simultaneously
- The player taking hits chooses which pieces to remove
- Warriors must be removed before buildings or tokens
- Players must remove all their warriors in the clearing before removing other pieces

Some effects can allow extra hits during battle.
```

### Example 2
Input
```bash
python -m query_db "What are some effects that can cause extra hits during battle?"
```

Ouput
```bash
Here are some effects that can cause extra hits during battle:

1. Defenseless Defender: If the defender has no warriors in the clearing of battle, the attacker deals an extra hit.

2. Brutal Tactics Card: When attacking, the player may deal an extra hit, but the defender scores one point.

3. Ambush Cards: The defender can play an ambush card matching the clearing's suit to deal two immediate hits, which are not limited by the number of warriors.

These extra hits are not constrained by the number of warriors in the clearing and can be dealt by a single warrior.
```

---
## Setup

This project is designed to run in a Python virtual environment (`venv`). Follow these steps to set it up:

1. Install `virtualenv` package
```bash
pip install virtualenv
```

2. **Create a virtual environment**:

```bash
virtualenv venv
```

3. **Activate the virtual environment**:

- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

4. **Install dependencies**:

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