# Voice Hospital Finder Bot Documentation

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Architecture](#3-architecture)
    *   [State Management](#state-management)
    *   [Conversational Flow (LangGraph)](#conversational-flow-langgraph)
    *   [Core Tools](#core-tools)
    *   [Database](#database)
    *   [Utilities](#utilities)
4.  [Setup and Installation](#4-setup-and-installation)
5.  [Usage](#5-usage)
6.  [Configuration](#6-configuration)
7.  [Extensibility](#7-extensibility)
8.  [Troubleshooting](#8-troubleshooting)
9.  [Dependencies](#9-dependencies)

## 1. Introduction
The Voice Hospital Finder Bot is an AI-powered conversational agent designed to help users find nearby hospitals based on their voice commands. It leverages advanced natural language processing, speech-to-text, and text-to-speech technologies to provide a seamless voice-controlled experience. The bot can understand user queries related to hospital types, insurance providers, and location, and then provide a ranked list of relevant hospitals.

## 2. Features
*   **Voice Interface**: Interact with the bot using natural spoken language.
*   **Speech-to-Text (STT)**: Transcribes user's spoken queries into text.
*   **Natural Language Understanding (NLU)**: Extracts key entities like location, hospital type, and insurance provider from text, using either spaCy (default) or an optional LLM for more nuanced intent and entity recognition.
*   **Text-to-Speech (TTS)**: Converts bot's responses into natural-sounding speech, with an optional LLM-driven dialogue refinement for more natural conversational flow.
*   **Hospital Search**: Finds hospitals based on user-specified criteria (location, type, insurance).
*   **Location-Based Services**: Ranks hospitals by proximity to the user's specified location.
*   **Conversational State Management**: Maintains context throughout the conversation, utilizing a clarifier to validate recognized entities (like location) and prompt the user for missing information.
*   **Database Integration**: Stores and retrieves hospital information from an SQLite database.
*   **Configurable**: Easy to adjust parameters for language models, search criteria, and conversational flow.

## 3. Architecture
The application is built around a `langgraph` StateGraph, which orchestrates the conversational flow. It integrates various AI models and custom tools to process voice input, understand user intent, lookup information, and generate spoken responses.

You can view the architectural flowchart [here](https://excalidraw.com/#json=oW09Ru21JJKzaYMnwFm-C,1YmWJCb3bafyPVmYW73rCQ).

### State Management
The `HospitalFinderState` (defined in `db/models.py`) is a Pydantic model that holds the entire state of the conversation. It tracks:
*   `uid`: Unique identifier for the session.
*   `input_audio_path`: Path to the last recorded user audio.
*   `transcription`: Result of the STT process.
*   `recognition`: Structured information extracted from the user's query (intent, location, hospital type, insurance).
*   `clarify_user_response_audio_path`, `clarify_transcription`, `clarify_recognition`: Temporary fields for clarification turns.
*   `hospitals_found`: List of hospitals returned by the lookup tool.
*   `clarify_bot_response_audio_path`: Path to the bot's audio response during clarification.
*   `turn_count`: Number of conversational turns.
*   `last_question`: The last question asked by the bot.
*   `final_response`: The final text/audio response to the user.
*   `final_response_audio_path`: Path to the final audio response.
*   `user_wants_exit`: A flag to indicate if the user wants to end the conversation.

### Conversational Flow (LangGraph)
The core logic resides in `graphs/hospital_graph.py`, which defines a `StateGraph` with several nodes and conditional edges:

**Nodes:**
*   **`record_transcribe_recognize`**:
    *   Records user's voice input.
    *   Transcribes the audio using `transcribe_audio_tool`.
    *   Checks for exit keywords.
    *   Recognizes entities and intent using `recognize_query_tool`.
    *   Updates the `HospitalFinderState` with transcription and recognition results.
*   **`clarifier`**:
    *   Activated if essential information (like location) is missing or needs validation.
    *   Generates and speaks a clarifying question using `text_to_speech_tool` to confirm or gather missing details.
    *   Manages `MAX_TURNS` to prevent infinite loops during clarification.
*   **`find_hospitals`**:
    *   Initiates the hospital search using `hospital_lookup_tool` based on the recognized location, hospital types, and insurance providers.
    *   Stores the results in `state.hospitals_found`.
*   **`generate_response`**:
    *   Constructs a natural language response based on `hospitals_found` (or a "no hospitals found" message).
    *   Converts the response to speech using `text_to_speech_tool` and plays it.
    *   Asks the user if they have another query.
    *   Summarizes and saves the conversation state.
    *   Resets the state for a new query, retaining the `uid`.

**Edges:**
*   **Normal Edges**: Define the sequential flow between `find_hospitals` -> `generate_response` and `generate_response` -> `record_transcribe_recognize`.
*   **Conditional Edges**:
    *   From `record_transcribe_recognize`: Directs to `clarifier` if the user doesn't want to exit, or `END` if exit keywords are detected.
    *   From `clarifier`: Directs back to `record_transcribe_recognize` for an attempt at collecting missing information, to `find_hospitals` if location is now found, to `generate_response` if `MAX_TURNS` is reached, or `END` if the user wants to exit.

### Core Tools
These tools are defined in `graphs/graph_tools.py` and implement specific functionalities:
*   **`transcribe_audio_tool`**:
    *   **Purpose**: Converts an audio file into transcribed text.
    *   **Implementation**: Utilizes `tools/transcribe.py` (likely uses OpenAI Whisper).
*   **`recognize_query_tool`**:
    *   **Purpose**: Extracts structured data (intent, location, hospital type, insurance) from a given query text.
    *   **Implementation**: Utilizes `tools/recognize.py` (likely uses a Language Model like Google Gemini or spaCy NER).
*   **`text_to_speech_tool`**:
    *   **Purpose**: Converts text into an audio file and returns its path.
    *   **Implementation**: Utilizes `tools/text_to_speech.py` (likely uses OpenAI TTS API).
*   **`hospital_lookup_tool`**:
    *   **Purpose**: Searches the database for hospitals matching specified criteria (user coordinates, hospital types, insurance providers).
    *   **Implementation**: Utilizes `tools/hospital_lookup.py`, which performs distance calculations (Haversine), filtering, and scoring based on rating and distance.

### Database
The application uses an SQLite database (`data/hospitals.sqlite`) managed by `Pony ORM`.
*   **Model**: The `Hospital` entity (defined in `db/db.py`) stores detailed information about hospitals: `hospital_id`, `hospital_name`, `location`, `latitude`, `longitude`, `address`, `hospital_type` (comma-separated string), `insurance_providers` (comma-separated string), and `rating`.
*   **Initialization**: `db/db.py` creates the database file and table, and if the database is empty, it populates it with synthetic hospital records generated by `db/hospital_generator.py`.

### Utilities
The `utils/utils.py` module provides helper functions:
*   `record_audio`: Records audio from the microphone.
*   `play_audio`: Plays an audio file.
*   `save_state`: Saves the current conversational state.
*   `summarize_conversation`: Summarizes the ongoing dialogue.

## 4. Setup and Installation

### Prerequisites
*   Python 3.8+
*   `ffmpeg`: Required for audio processing. (Install via your system's package manager, e.g., `sudo apt-get install ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS, or download binaries for Windows).
*   spaCy English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/navinpandiyan/voice-hospital-finder-bot.git
    cd voice-hospital-finder-bot
    ```
2.  **Install dependencies and run the application using `uv`:**
    If you have `uv` installed, simply run:
    ```bash
    uv run app.py
    ```
    `uv` will automatically create and manage a virtual environment and install dependencies from `requirements.txt`.

    Alternatively, if you prefer `pip` and manual virtual environments:
    2.  **Create and activate a virtual environment:**
        ```bash
        python -m venv venv
        # On Windows
        .\venv\Scripts\activate
        # On macOS/Linux
        source venv/bin/activate
        ```
    3.  **Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
    4.  **Set up Environment Variables**: Create a `.env` file in the root directory and add the necessary API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_for_whisper_and_tts"
    LLM_API_KEY="your_llm_api_key_for_intent_identification" # For models configured in settings/config.py (e.g., Google Gemini)
    LLM_BASE_URL="your_llm_base_url" # Optional: e.g., for local LLMs or specific API endpoints
    ```
    * `OPENAI_API_KEY` is required for Speech-to-Text (Whisper) and Text-to-Speech functions.
    * `LLM_API_KEY` and `LLM_BASE_URL` are used for intent identification and query recognition if using models other than OpenAI, as configured in `settings/config.py`.

### Database Initialization
The SQLite database (`data/hospitals.sqlite`) will be automatically created and populated with synthetic data on the first run of `app.py` if it doesn't already exist or is empty.

## 5. Usage

To start the Voice Hospital Finder Bot, run the `app.py` script:
```bash
python app.py
```

The bot will then prompt you verbally. You can then speak your queries, such as:
*   "Find a cardiology hospital in Dubai."
*   "Are there any hospitals that accept Aetna insurance in Abu Dhabi?"
*   "Show me hospitals for pediatrics near Al Ain."
*   "I am looking for a hospital in Sharjah." (If location is missing, the bot will ask for clarification).

### Exit Keywords
You can end the conversation at any point by saying: "no", "nope", "stop", "exit", "quit", "end", "that's all", "done", "nothing", "bye", or "goodbye".

## 6. Configuration
The `settings/config.py` file contains various configurable parameters:

*   **`NLP_MODEL`**: spaCy model for Named Entity Recognition (default: `en_core_web_sm`).
*   **`LOGGER`**: Logging configuration.
*   **`CITY_COORDINATES`**: Dictionary of city names and their geographic coordinates (used for data generation and potential recognition fallback).
*   **`HOSPITAL_TYPES`**: List of supported hospital specialties/types.
*   **`INSURANCE_PROVIDERS`**: List of supported insurance providers.
*   **`FUZZY_MATCH_THRESHOLD`**: Threshold for fuzzy matching of entities.
*   **Speech-to-Text (`TRANSCRIBER_OPENAI_MODEL`, `TRANSCRIBER_LANGUAGE`)**: OpenAI Whisper model and language.
*   **Recognizer (`RECOGNIZER_MODEL`, `RECOGNIZER_TEMPERATURE`)**: Configures the model for query recognition. By default, spaCy is used for entity extraction. Setting this to an LLM enables LLM-based intent and entity recognition.
*   **Clarifier (`CLARIFIER_MODEL`, `CLARIFIER_TEMPERATURE`)**: Language model for generating clarifying questions.
*   **Text-to-Dialogue (`TEXT_TO_DIALOGUE_MODEL`, `TEXT_TO_DIALOGUE_TEMPERATURE`)**: Configures the model for generating natural dialogue. Utilizing an LLM here refines the bot's responses, making them more conversational and contextually aware.
*   **`HOSPITAL_DATA_FOLDER`, `HOSPITAL_DATA_FILE_NAME`**: Paths for the hospital database.
*   **`MAX_TURNS`**: Maximum number of conversation turns before the bot gives up on a missing piece of information (e.g., location).

## 7. Extensibility
Developers can extend the bot's functionality by:
*   **Adding new tools**: Create new Python functions decorated with `@tool` and integrate them into `graphs/graph_tools.py` and the `hospital_finder_graph`.
*   **Modifying the conversational flow**: Adjust nodes and edges in `graphs/hospital_graph.py` to change how the bot interacts.
*   **Enhancing entity recognition**: Update `tools/recognize.py` or `settings/config.py` to support more entities or refine existing ones.
*   **Integrating different LLMs/APIs**: Change the model names in `settings/config.py` to use other STT, NLU, or TTS services.
*   **Customizing hospital data**: Modify `db/hospital_generator.py` to generate different types of synthetic data or integrate with a real hospital data source.
*   **Improving scoring/filtering**: Adjust the logic in `tools/hospital_lookup.py` to change how hospitals are ranked or filtered.

## 8. Troubleshooting
*   **`spaCy model 'en_core_web_sm' not found`**: Run `python -m spacy download en_core_web_sm`.
*   **`ffmpeg` not found**: Ensure `ffmpeg` is installed and accessible in your system's PATH.
*   **API Key issues**: Verify that `OPENAI_API_KEY` (and any other necessary API keys) are correctly set in your `.env` file.
*   **No hospitals found**: Check if the database (`data/hospitals.sqlite`) is populated. If not, it should be auto-generated on first run. Also, ensure your query's location, hospital types, and insurance providers match the data.
*   **Bot not responding**: Check the console for any error messages in the logs generated by `settings/config.LOGGER`.

## 9. Dependencies
The `requirements.txt` file lists all package dependencies. Key dependencies include:
*   `langchain`, `langgraph`: For AI orchestration and conversational flow.
*   `openai`, `openai-whisper`: For Speech-to-Text and Text-to-Speech.
*   `spacy`: For Named Entity Recognition (NLU).
*   `ponyorm`: For database interaction (SQLite).
*   `pandas`, `numpy`: For data manipulation in hospital lookup.
*   `geopy`: For geographical calculations (though Haversine is custom implemented).
*   `pydantic`: For data validation and settings management.
*   `python-dotenv`: For loading environment variables.
