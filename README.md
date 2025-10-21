# Hospital Finder Bot

This project implements a voice-enabled hospital finder bot. It leverages a combination of speech-to-text, natural language understanding, Retrieval Augmented Generation (RAG) with FAISS vector database, LLM fine-tuning, and text-to-speech to provide an interactive experience for users looking for hospitals.

## Key Features

*   🎤 **Voice Interaction**: Speech → text → structured intent → response → speech.
*   🧠 **Intent Understanding**: Extracts user intent, location, insurance provider, and hospital names.
*   🧾 **Structured Knowledge Base**: Uses synthetic hospital + insurance plan data.
*   ⚡ **FAISS Vector Search**: Enables semantic retrieval of relevant hospitals.
*   🧩 **Extensible Architecture**: Pluggable with new LLM, STT, and TTS models.
*   🔄 **Domain Fine-Tuning**: Supports LoRA fine-tuning using PEFT for healthcare contexts.
*   🗣️ **Natural Conversation**: Generates context-aware, human-like dialogue.

## Architecture & Flow
1. **Input Layer (Voice/Text)**: User speaks or types a query.
2. **Intent Recognition**: The system identifies one of six intents:
   - Find Nearest
   - Find Best
   - Find by Insurance
   - Find by Hospital
   - Compare Hospitals
   - Exit
3. **Follow-up Clarification**: For certain intents, the bot asks for clarification (e.g., missing location).
4. **Information Retrieval**: A **RAG pipeline** searches the FAISS vector database for relevant hospital or insurance data.
5. **Response Generation**: The LLM generates a structured, natural response.
6. **Loop Back**: Accept another query or Exit.
7. **Text-to-Speech**: The final response is optionally converted to audio output.

## Main Configurables
Key configurable parameters define the bot’s behavior and operating mode:

| Parameter | Description | Example / Default |
|-----------|-------------|-----------------|
| `MODE` | Running mode — text chatbot or voice-enabled bot | `"chatbot"` (`"chatbot"` or `"voicebot"`) |
| `TRANSCRIBER_OPENAI_MODEL` | Speech-to-Text model for audio input | `"whisper-1"` |
| `TRANSCRIBER_LANGUAGE` | Language for transcription | `"en"` |
| `USE_LLM_FOR_RECOGNITION` | If True, uses LLM for intent recognition instead of spaCy | `TRUE` |
| `RECOGNIZER_MODEL` | LLM for detecting intent, entities, and query structure | `"google/gemini-2.0-flash-001"` |
| `RECOGNIZER_TEMPERATURE` | Sampling temperature for recognition LLM | `0.1` |
| `CLARIFIER_MODEL` | Model for clarifying follow-ups | `"google/gemini-2.0-flash-001"` |
| `CLARIFIER_TEMPERATURE` | Temperature setting for clarification model | `0.1` |
| `TEXT_TO_DIALOGUE` | Converts factual responses to natural dialogue before TTS | `FALSE` |
| `TEXT_TO_DIALOGUE_MODEL` | Model for dialogue conversion | `"google/gemini-2.0-flash-001"` |
| `TEXT_TO_DIALOGUE_TEMPERATURE` | Temperature for dialogue conversion | `0.1` |
| `MAX_TURNS` | Maximum dialogue turns for a session | `10` |
| `LOOKUP_MODE` | Mode of hospital lookup (`simple` or `rag`) | `"rag"` |
| `RAG_GROUNDER_MODEL` | Model used for RAG grounding | `"google/gemini-2.0-flash-001"` |
| `RAG_GROUNDER_TEMPERATURE` | Temperature for RAG grounding model | `0.1` |
| `GROUND_WITH_FINE_TUNE` | If True and intent is "find by hospital," use fine-tuned QLoRA model | `TRUE` |

---

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
