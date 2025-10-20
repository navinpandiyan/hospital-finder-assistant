# Voice Hospital Finder Bot

This project implements a voice-enabled hospital finder bot. It leverages a combination of speech-to-text, natural language understanding, Retrieval Augmented Generation (RAG) with FAISS vector database, LLM fine-tuning, and text-to-speech to provide an interactive experience for users looking for hospitals.

## Features

*   **Voice Interface**: Interact with the bot using spoken language (transcription and text-to-speech).
*   **Natural Language Understanding**: Extracts user intent, location, hospital types, and insurance providers from natural language queries.
*   **Hospital Database**: Stores structured information about hospitals and insurance plans using SQLite and Pony ORM.
*   **FAISS Vector Database**: Enables efficient semantic search for hospitals based on query embeddings.
*   **Retrieval Augmented Generation (RAG)**: Combines vector search with large language models (LLMs) to provide grounded responses.
*   **LLM Fine-tuning (QLoRA)**: Optionally fine-tunes a language model to improve response generation for specific use cases (e.g., insurance queries).
*   **Modular Design**: Structured into `db/`, `graphs/`, and `tools/` directories for clear separation of concerns.

## Project Structure

*   `app.py`: Main application entry point, initializes the RAG retriever and starts the LangGraph conversational flow.
*   `db/`: Contains database-related functionalities.
    *   `db.py`: Sets up the SQLite database, defines `Hospital` and `InsurancePlan` models, populates data, creates the FAISS vector database, and orchestrates LLM fine-tuning data generation and training.
    *   `models.py`: Defines Pydantic models for structured data, including LLM responses, RAG grounding, and the overall `HospitalFinderState`.
    *   `modules/`: Sub-directory for database-related modules like data generation and fine-tuning.
        *   `fine_tuner.py`: Implements the QLoRA fine-tuning process for a language model.
        *   `hospital_generator.py`: (Not read, but inferred from `db.py`) Likely generates synthetic hospital data.
        *   `insurance_generator.py`: (Not read, but inferred from `db.py`) Likely generates synthetic insurance plan data.
        *   `vector_db_generator.py`: (Not read, but inferred from `db.py`) Likely handles the creation of the FAISS vector database.
        *   `fine_tune_data_generator.py`: (Not read, but inferred from `db.py`) Likely generates data for LLM fine-tuning.
*   `graphs/`: Contains the conversational graph logic.
    *   `hospital_graph.py`: (Not read, but inferred from `app.py`) Likely defines the LangGraph state machine for the hospital finding process.
    *   `graph_tools.py`: Defines Langchain `tool`s used within the conversational graph, such as `transcribe_audio_tool`, `recognize_query_tool`, `text_to_speech_tool`, `hospital_lookup_tool`, and `hospital_lookup_rag_tool`.
*   `tools/`: Houses various utility tools.
    *   `rag_retrieve.py`: Implements the `HospitalRAGRetriever` class for retrieving hospital information using FAISS and grounding responses with an LLM (optionally fine-tuned).
    *   `transcribe.py`: (Not read, but inferred from `graph_tools.py`) Likely handles speech-to-text functionality.
    *   `recognize.py`: (Not read, but inferred from `graph_tools.py`) Likely handles natural language recognition and extraction.
    *   `text_to_speech.py`: (Not read, but inferred from `graph_tools.py`) Likely handles text-to-speech functionality.
    *   `hospital_lookup.py`: (Not read, but inferred `graph_tools.py`) Likely contains the logic for direct hospital database lookup.
*   `settings/`: Configuration files.
    *   `config.py`: Centralized configuration for the application, including API keys, model names, hyperparameters, and file paths.
    *   `prompts.py`: (Not read, but inferred from `rag_retrieve.py`) Likely contains LLM prompts used for various tasks.
    *   `client.py`: (Not read, but inferred from `rag_retrieve.py`) Likely sets up the LLM client.

## 🚀 Setup and Installation (using `uv`)

### 1. **Clone the repository**
```bash
git clone https://github.com/navinpandiyan/voice-hospital-finder-bot.git
cd voice-hospital-finder-bot
```

### 2. Install uv (Modern Python Package Manager)

If you don’t have uv installed, run:

```bash
pip install uv
```

Or, for a faster global installation:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify installation:

```bash
uv --version
```

### 3. Create a virtual environment and sync dependencies

`uv` automatically manages your virtual environment and dependencies.

```bash
uv sync
```

This will:

*   Create a `.venv/` environment
*   Install dependencies from `pyproject.toml` or `requirements.txt`
*   Ensure consistent versions across systems

### 4. Set up environment variables

Create a `.env` file in the project root and add your API keys:

```
OPENAI_API_KEY="your_openai_api_key_here"  # For TTS and Embedding Models
LLM_API_KEY="your_llm_api_key_here"        # For Backend LLM (OpenAI/OpenRouter/etc)
LLM_BASE_URL="https://api.openai.com/v1"   # Or custom LLM endpoint
```

### 5. Run the application
```bash
uv run app.py
```

The application will:

*   Initialize the database (if empty)
*   Generate synthetic data
*   Create the FAISS vector database
*   (Optionally) fine-tune the LLM

Launch the voice bot 🎙️
