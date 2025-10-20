# Source Code Documentation for Voice Hospital Finder Bot

This document provides a detailed overview of the key Python modules, classes, and functions within the Voice Hospital Finder Bot project. It aims to clarify the architecture, component interactions, and functional responsibilities of the codebase.

## Table of Contents
- [Project Structure](#project-structure)
- [Application Entry Point](#application-entry-point)
    - [app.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/app.py)
- [Database Management](#database-management)
    - [db/models.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/db/models.py)
    - [db/db.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/db/db.py)
    - [db/hospital_generator.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/db/hospital_generator.py)
    - [db/vector_db_generator.py](#dbvector_db_generator.py-module)
    - [db/fine_tune.py](#dbfine_tune.py-module)
- [Graph Orchestration](#graph-orchestration)
    - [graphs/hospital_graph.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/graphs/hospital_graph.py)
    - [graphs/graph_tools.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/graphs/graph_tools.py)
- [Settings and Configuration](#settings-and-configuration)
    - [settings/config.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/settings/config.py)
    - [settings/prompts.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/settings/prompts.py)
- [Tools](#tools)
    - [tools/hospital_lookup.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/tools/hospital_lookup.py)
    - [tools/recognize.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/tools/recognize.py)
    - [tools/rag_retrieve.py](#toolsrag_retrieve.py-module)
    - [tools/record.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/tools/record.py)
    - [tools/text_to_speech.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/tools/text_to_speech.py)
    - [tools/transcribe.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/tools/transcribe.py)
- [Utilities](#utilities)
    - [utils/utils.py](https://github.com/navinpandiyan/voice-hospital-finder-bot/blob/main/utils/utils.py)

---

## Project Structure
The project is organized into several directories, each serving a specific purpose in the Voice Hospital Finder Bot's architecture.

*   **`/` (Root Directory)**:
    *   `app.py`: The main entry point for the application.
    *   `README.md`: General project information and setup instructions.
    *   `requirements.txt`: Python dependencies for the project.
    *   `pyproject.toml`: Project configuration file.
    *   `.gitignore`: Specifies intentionally untracked files to ignore.
*   **`audios/`**:
    *   Contains subdirectories for input and output audio files (`input/`, `output/`) generated during the conversation.
*   **`data/`**:
    *   Stores static data or database files, such as `hospitals.sqlite`.
    *   `data/rag_llm/`: Contains the fine-tuned QLoRA model checkpoints and tokenizer.
    *   `data/vdb_hospitals/`: Contains the FAISS vector database index files.
*   **`db/`**:
    *   `db.py`: Database connection and setup for SQLite.
    *   `hospital_generator.py`: Script for generating dummy hospital data.
    *   `models.py`: Pydantic models defining the state and LLM output structures.
    *   `modules/fine_tune.py`: Module for fine-tuning the QLoRA LLM.
    *   `modules/vector_db_generator.py`: Module for creating the FAISS vector db.
*   **`docs/`**:
    *   `DOCUMENTATION.md`: High-level project documentation.
    *   `SOURCE_CODE_DOCUMENTATION.md`: This file, detailing the codebase (currently being generated).
    *   `hospital_finder_flow.png`: Architectural flowchart.
*   **`graphs/`**:
    *   `hospital_graph.py`: Defines the LangGraph state machine for the conversational flow.
    *   `graph_tools.py`: Wrappers for external tools used within the graph.
*   **`outputs/`**:
    *   Stores outputs such as conversation summaries and state snapshots.
*   **`settings/`**:
    *   `client.py`: Configuration and initialization for external API clients (e.g., OpenAI, LLM).
    *   `config.py`: Centralized configuration variables and constants.
    *   `prompts.py`: Defines system and user prompts for LLM interactions.
*   **`tools/`**:
    *   `hospital_lookup.py`: Logic for searching and filtering hospitals from the SQLite DB.
    *   `recognize.py`: Handles Natural Language Understanding (NLU) for entity extraction.
    *   `rag_retrieve.py`: Implements Retriever-Augmented Generation (RAG) using the vector database and QLoRA LLM.
    *   `record.py`: Manages audio recording from the microphone.
    *   `text_to_speech.py`: Converts text into spoken audio.
    *   `transcribe.py`: Transcribes audio to text.
*   **`utils/`**:
    *   `utils.py`: General utility functions (e.g., audio playback, geocoding, state saving).

## Application Entry Point
### `app.py` Module

This module serves as the primary entry point for the Voice Hospital Finder Bot application. It initializes the conversational state and orchestrates the execution of the `hospital_finder_graph` to manage the dialog flow.

**Functions:**

*   **`async def main()`**:
    *   **Purpose**: Initiates a hospital finder session. It creates an initial `HospitalFinderState` and asynchronously invokes the `hospital_finder_graph` to process user interactions and generate responses.
    *   **Logic**:
        1.  Initializes `HospitalFinderState` to maintain the session's context.
        2.  Logs the start of the hospital finder session.
        3.  Calls `hospital_finder_graph.ainvoke(state)` to run the conversational graph.
    *   **Usage**: Designed to be run as an asynchronous application.

## Database Management
### `db/models.py` Module

This module defines the Pydantic models used for managing the state of the hospital finder session and structuring LLM outputs and TTS responses.

**Classes:**

*   **`LLMResponseModel`**:
    *   **Purpose**: Defines the expected structure for responses generated by Language Model (LLM) calls, particularly for intent and entity recognition.
    *   **Fields**:
        *   `intent` (str): The recognized intent, defaulting to "find_hospital".
        *   `location` (Optional[str]): The recognized location entity.
        *   `hospital_type` (List[str]): A list of recognized hospital types.
        *   `insurance` (List[str]): A list of recognized insurance providers.
        *   `n_hospitals` (Optional[int]): The number of hospitals to search for.
        *   `distance_km` (Optional[Union[int, float]]): The search radius in kilometers.

*   **`RAGGroundedResponseModel`**:
    *   **Purpose**: Defines the expected structure for responses generated after RAG-based grounding, combining identified hospital IDs with a generated dialogue.
    *   **Fields**:
        *   `hospital_ids` (List[int]): A list of `hospital_id`s identified as relevant for the response.
        *   `dialogue` (str): The generated natural language response based on the RAG process.

*   **`HospitalFinderState`**:
    *   **Purpose**: Represents the complete state of a single conversational session with the hospital finder bot. This model is crucial for maintaining context throughout the LangGraph flow.`
    *   **Fields**:
        *   `uid` (str): A unique identifier for the session, generated automatically.
        *   `input_audio_path` (Optional[str]): Path to the recorded user input audio.
        *   `transcription` (Optional[dict]): Transcription of the user's input audio.
        *   `recognition` (Optional[dict]): NLU recognition results (e.g., intent, entities).
        *   `clarify_user_response_audio_path` (Optional[str]): Path to user audio when clarifying.
        *   `clarify_transcription` (Optional[dict]): Transcription of clarification audio.
        *   `clarify_recognition` (Optional[dict]): NLU recognition results for clarification.
        *   `hospitals_found` (Optional[List[dict]]): List of hospitals matching criteria.
        *   `clarify_bot_response_audio_path` (Optional[str]): Path to bot audio for clarification.
        *   `turn_count` (int): Counter for the number of turns in the conversation.
        *   `last_question` (Optional[str]): The last question asked by the bot, used for clarifying.
        *   `final_response` (Optional[dict]): The final structured response from the bot.
        *   `final_response_audio_path` (Optional[str]): Path to the final response audio.
        *   `user_wants_exit` (bool): Flag indicating if the user wishes to end the session.

*   **`TTSResponseModel`**:
    *   **Purpose**: Defines the structure for Text-to-Speech (TTS) module responses, including the dialogue and its desired tone.
    *   **Fields**:
        *   `dialogue` (str): The text content to be converted to speech.
        *   `tone` (Optional[str]): The emotional tone for the TTS output (e.g., 'friendly', 'informative').

### `db/db.py` Module

This module is responsible for setting up the SQLite database using Pony ORM and initializing it with synthetic hospital data if the database is empty.

**Database Configuration:**

*   **`db_folder`**: The directory specified by `HOSPITAL_DB_FOLDER` (e.g., `data/`) where the database file will be stored.
*   **`db_file`**: The absolute path to the SQLite database file (e.g., `data/hospitals.sqlite`). The directory is created if it doesn't exist.
*   **`db`**: A `pony.orm.Database` instance bound to the SQLite file, configured to create the database if it doesn't exist.

**Class:**

*   **`Hospital(db.Entity)`**:
    *   **Purpose**: Defines the database schema for a hospital entity using Pony ORM.
    *   **Fields**:
        *   `hospital_id` (int): Unique identifier for the hospital.
        *   `hospital_name` (str): Name of the hospital.
        *   `location` (str): General location (city/region).
        *   `latitude` (float): Geographic latitude.
        *   `longitude` (float): Geographic longitude.
        *   `address` (str): Full street address.
        *   `hospital_type` (str): Comma-separated list of types/specialties.
        *   `insurance_providers` (str): Comma-separated list of accepted insurance providers.
        *   `rating` (float): Hospital rating.

**Initialization Process:**

*   **`db.generate_mapping(create_tables=True)`**: Generates the database tables based on the `Hospital` entity definition.
*   **Database Population**:
    1.  Uses a `db_session` to check if the `Hospital` table is empty.
    2.  If empty, it calls `db.hospital_generator.generate_hospital_records` to create synthetic hospital data.
    3.  Iterates through the generated records and creates `Hospital` entities, populating the database.
    4.  Logs the number of hospitals generated.

### `db/hospital_generator.py` Module

This module is responsible for generating synthetic hospital training data, which includes realistic hospital names, locations, types, insurance providers, ratings, and addresses. This data is used to populate the application's database for demonstration and testing purposes.

**Constants:**

*   **`HOSPITAL_SUFFIXES` (list)**: A list of common suffixes for hospital names (e.g., "Hospital", "Medical Center").
*   **`CITY_ADDRESSES` (dict)**: A dictionary mapping cities to lists of street names within those cities, providing realistic address components.

**Functions:**

*   **`def generate_hospital_records(num_hospitals=150) -> list`**:
    *   **Purpose**: Generates a specified number (`num_hospitals`) of synthetic hospital records.
    *   **Parameters**:
        *   `num_hospitals` (int): The total number of hospital records to generate. Defaults to 150.
    *   **Process**:
        1.  Iterates `num_hospitals` times, generating details for each hospital.
        2.  Randomly selects a `city` from `settings.config.CITY_COORDINATES` and generates slightly varied `latitude` and `longitude` around the city's base coordinates.
        3.  Randomly selects 1 to 3 `hospital_type` (specialties) and `insurance_providers` from `settings.config` lists.
        4.  Constructs a `hospital_name` using the city, a selection of specialties, and a random `HOSPITAL_SUFFIX`.
        5.  Assigns a random `rating` between 1.0 and 5.0.
        6.  Generates a realistic `address` by combining a random street number, a street name pertinent to the chosen city (from `CITY_ADDRESSES`), and the city name.
        7.  Appends the structured hospital record to a list.
    *   **Returns**: A `list` of dictionaries, where each dictionary represents a complete synthetic hospital record.


### `db/modules/vector_db_generator.py` Module

This module is responsible for creating and saving a FAISS vector database from hospital records. This vector database is a crucial component for the Retriever-Augmented Generation (RAG) system, enabling semantic search and retrieval of relevant hospital information.

**Functions:**

*   **`def create_vector_db_from_records(hospital_records, embedding_model=None) -> FAISS`**:
    *   **Purpose**: Builds a FAISS vector database from a list of hospital records and saves it locally.
    *   **Parameters**:
        *   `hospital_records` (list): A list of dictionaries, where each dictionary represents a hospital record with fields like `hospital_name`, `address`, `location`, `hospital_type`, `insurance_providers`, `rating`, `latitude`, `longitude`.
        *   `embedding_model` (Optional[EmbeddingModel]): An optional pre-initialized embedding model (e.g., `OpenAIEmbeddings`). If `None`, `OpenAIEmbeddings` is used by default.
    *   **Process**:
        1.  Initializes the embedding model.
        2.  Iterates through each `hospital_record`:
            *   Constructs a `content` string that semantically describes the hospital, emphasizing key attributes for retrieval.
            *   Creates a `metadata` dictionary containing all original hospital record fields, including `hospital_id`.
            *   Appends a `Document` object (from `langchain.docstore.document`) with the `content` and `metadata` to a list of documents.
        3.  Builds the FAISS vector database from these `Document` objects using the specified `embedding_model`.
        4.  Saves the generated FAISS vector database locally to the path defined by `settings.config.VECTOR_DB_FOLDER`.
    *   **Returns**: The created `FAISS` vector database instance.

### `db/modules/fine_tune.py` Module

This module provides the functionality to fine-tune a Large Language Model (LLM) using QLoRA (Quantized Low-Rank Adaptation) for specific tasks, such as enhancing insurance query understanding and response generation. It leverages the `transformers` and `peft` libraries for efficient fine-tuning.

**Constants & Configuration:**

*   **`BASE_MODEL` (str)**: The base pre-trained language model to be fine-tuned (e.g., "tiiuae/falcon-rw-1b").
*   **`TOKENIZER_MODEL` (str)**: The tokenizer model, usually the same as `BASE_MODEL`.
*   **`FINE_TUNE_OUTPUT_DIR` (str)**: Directory to save the fine-tuned model and tokenizer (e.g., "data/rag_llm").
*   **Training Hyperparameters**: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `MAX_SEQ_LEN`, `GRADIENT_ACCUMULATION_STEPS`, `FP16`.
*   **Logging / Checkpoints**: `SAVE_STEPS`, `LOGGING_STEPS`.
*   **LoRA / QLoRA Config**: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `TARGET_MODULES`, `LORA_TASK_TYPE`.

**Functions:**

*   **`def fine_tune_insurance_llm(data_path: str = "data/insurance_data.json") -> str`**:
    *   **Purpose**: Orchestrates the entire QLoRA fine-tuning process for an LLM on insurance-related data.
    *   **Parameters**:
        *   `data_path` (str): Path to the JSON dataset used for fine-tuning. This dataset should contain `instruction`, `context`, and `response` fields.
    *   **Process**:
        1.  Loads the raw data from the specified `data_path` and converts it into a `datasets.Dataset` object.
        2.  Formats the dataset into instruction-response prompts suitable for causal language modeling.
        3.  Loads the base model and its tokenizer.
        4.  Prepares the model for k-bit training (QLoRA) to enable efficient fine-tuning on consumer hardware.
        5.  Configures LoRA parameters (rank, alpha, dropout, target modules) and applies them to the base model using `get_peft_model`.
        6.  Tokenizes the formatted dataset.
        7.  Sets up `TrainingArguments` (output directory, batch size, learning rate, etc.) and `Trainer`.
        8.  Starts the fine-tuning process (`trainer.train()`).
        9.  Saves the fine-tuned PEFT model adapter and tokenizer to `FINE_TUNE_OUTPUT_DIR`.
    *   **Returns**: The path to the directory where the fine-tuned model was saved.

## Graph Orchestration
### `graphs/hospital_graph.py` Module

This module defines and orchestrates the conversational flow of the Voice Hospital Finder Bot using LangGraph. It sets up a state machine with various nodes representing different stages of user interaction and system processing.

**Graph Structure:**

The `hospital_finder_graph` is a `StateGraph` that manages the `HospitalFinderState`. It consists of the following nodes and conditional edges:

**Nodes:**

*   **`async def record_transcribe_recognize(state: HospitalFinderState)`**:
    *   **Purpose**: Handles user audio input, transcribes it, and performs Natural Language Understanding (NLU) to recognize intent and entities.
    *   **Inputs**: `HospitalFinderState` (current state of the conversation).
    *   **Process**:
        1.  Generates a unique ID (`uid`) if not already present in the state.
        2.  Records user audio for a set duration.
        3.  Transcribes the audio to text using `transcribe_audio_tool`.
        4.  Checks the transcribed text for exit keywords to allow early termination.
        5.  Recognizes intent and entities (e.g., location, hospital type, insurance) from the transcribed text using `recognize_query_tool`.
        6.  Merges or updates the `recognition` data in the state, handling clarifications and preserving previously recognized entities.
        7.  Increments `turn_count`.
    *   **Outputs**: Updated `HospitalFinderState` with audio path, transcription, and recognition results.

*   **`async def clarifier(state: HospitalFinderState)`**:
    *   **Purpose**: Validates crucial entities (like `location`) and prompts the user for missing information if necessary. This acts as a feedback loop to ensure essential data is collected before proceeding.
    *   **Inputs**: `HospitalFinderState`.
    *   **Process**:
        1.  Checks if `location` is present in `state.recognition`. If found, bypasses clarification.
        2.  If `location` is missing and `turn_count` meets `MAX_TURNS`, sets an error and flags `user_wants_exit`.
        3.  Constructs a clarification prompt based on missing information and existing recognized entities.
        4.  Converts the prompt text to speech using `text_to_speech_tool` and plays the audio for the user.
    *   **Outputs**: Updated `HospitalFinderState`, potentially with a clarification prompt audio path.

*   **`async def find_hospitals(state: HospitalFinderState)`**:
    *   **Purpose**: Coordinates the hospital search. Depending on configurations (e.g., `LOOKUP_MODE`), this node either uses a direct hospital lookup or employs the RAG system to find and ground responses about hospitals.
    *   **Inputs**: `HospitalFinderState`.
    *   **Process**:
        1.  Ensures `location_coordinates` are available. If not, sets an error.
        2.  If `LOOKUP_MODE` is "rag":
            *   Initializes `HospitalRAGRetriever` (from `tools.rag_retrieve.py`).
            *   Passes user query parameters to `rag_search_wrapper` to retrieve and ground hospital information using the vector database and the fine-tuned QLoRA LLM.
            *   Updates `state.hospitals_found` and `state.final_response.dialogue` with results from the RAG process.
        3.  If `LOOKUP_MODE` is "simple":
            *   Uses `hospital_lookup_tool` to find hospitals near the specified coordinates, filtering by hospital types and insurance providers from the SQLite DB.
            *   Stores the found hospitals in `state.hospitals_found`.
    *   **Outputs**: Updated `HospitalFinderState` with a list of `hospitals_found` or an error, and potentially a generated dialogue from RAG.

*   **`async def generate_response(state: HospitalFinderState)`**:
    *   **Purpose**: Formulates the final or interim response to the user, converts it to speech, and manages the conversation flow for follow-up queries.
    *   **Inputs**: `HospitalFinderState`.
    *   **Process**:
        1.  If the RAG process in `find_hospitals` already generated a `dialogue`, it uses that directly. Otherwise, it constructs a natural language response based on `hospitals_found` (or a "no hospitals found" message).
        2.  Converts the response text to speech using `text_to_speech_tool` (optionally with LLM dialogue refinement).
        3.  Plays the audio response for the user.
        4.  Asks the user if they have any other queries, converting this follow-up question to speech and playing it.
        5.  Summarizes the conversation and saves the state for potential future analysis.
        6.  Resets the `HospitalFinderState` for a new query, preserving the `uid`.
    *   **Outputs**: Updated `HospitalFinderState` with `final_response` and `final_response_audio_path`.

**Conditional Edges:**

*   **From `record_transcribe_recognize`**:
    *   If `user_wants_exit` is `True`, transitions to `END`.
    *   Otherwise, transitions to `clarifier`.

*   **From `clarifier` (using `clarifier_conditional` function)**:
    *   If `user_wants_exit` is `True`, transitions to `END`.
    *   If `location` is found in `state.recognition`, transitions to `find_hospitals`.
    *   If `MAX_TURNS` is reached without a location, transitions to `generate_response` (to inform the user of the failure).
    *   If `location` is still missing and `MAX_TURNS` not reached, loops back to `record_transcribe_recognize` for re-prompting.

**Normal Edges:**

*   `find_hospitals` always transitions to `generate_response`.
*   `generate_response` always transitions back to `record_transcribe_recognize` (to allow for follow-up queries).

**Entry Point:**

*   The graph starts at the `record_transcribe_recognize` node, initiating the interaction by recording user input.

### `graphs/graph_tools.py` Module

This module defines specialized asynchronous tools, leveraging LangChain's `@tool` decorator, which are integrated into the `hospital_finder_graph`. These tools encapsulate specific functionalities such as audio transcription, natural language understanding, text-to-speech conversion, and hospital database lookup.

**Tools:**

*   **`async def transcribe_audio_tool(audio_path: str, uid: str) -> dict`**:
    *   **Purpose**: Transcribes spoken audio into text.
    *   **Parameters**:
        *   `audio_path` (str): The file path to the audio to be transcribed.
        *   `uid` (str): A unique identifier for the current session.
    *   **Returns**: A dictionary containing the `uid`, original `audio_path`, and the `transcribed_text`.
    *   **Internal Logic**: Internally calls `tools.transcribe.transcribe_wrapper`.

*   **`async def recognize_query_tool(query_text: str, uid: str, use_llm: bool = True) -> dict`**:
    *   **Purpose**: Extracts structured intent and entities (like location, hospital types, and insurance) from a given text query.
    *   **Parameters**:
        *   `query_text` (str): The text query to be processed.
        *   `uid` (str): A unique identifier for the current session.
        *   `use_llm` (bool): Flag to determine whether to use an LLM for recognition (default is `True`).
    *   **Returns**: A dictionary containing the `uid`, original `query`, recognized `intent`, `location`, `hospital_type` (list), and `insurance` (list).
    *   **Internal Logic**: Internally calls `tools.recognize.recognize_wrapper`.

*   **`async def rag_retrieve_tool(retriever: HospitalRAGRetriever, user_query: Optional[str] = None, ...) -> Tuple[List[dict], str]`**:
    *   **Purpose**: Performs a RAG-based search and grounding using the `HospitalRAGRetriever`.
    *   **Parameters**: Accepts various user input parameters like `user_query`, `user_loc`, `hospital_types`, `insurance_providers`, etc., along with an initialized `HospitalRAGRetriever`.
    *   **Returns**: A tuple containing a `List` of relevant hospital dictionaries and a generated `str` dialogue from the RAG process.
    *   **Internal Logic**: Internally calls `tools.rag_retrieve.rag_search_wrapper`. This tool is used within the `find_hospitals` node when `LOOKUP_MODE` is "rag".

*   **`async def text_to_speech_tool(text: str, uid: str, output_dir: str = "audios/output", convert_to_dialogue: bool = False) -> dict`**:
    *   **Purpose**: Converts a given text into an audio file using text-to-speech services.
    *   **Parameters**:
        *   `text` (str): The text content to be converted into speech.
        *   `uid` (str): A unique identifier for the current session.
        *   `output_dir` (str): The directory where the generated audio file will be saved.
        *   `convert_to_dialogue` (bool): Flag to enable LLM-driven dialogue refinement before TTS (default is `False`).
    *   **Returns**: A dictionary containing the `uid`, original `text`, the potentially modified `dialogue`, `tone`, the `audio_path` of the generated speech file, and a boolean indicating if an `llm_used` for dialogue conversion.
    *   **Internal Logic**: Internally calls `tools.text_to_speech.text_to_speech_wrapper`.

*   **`async def hospital_lookup_tool(user_lat: float, user_lon: float, hospital_types: Optional[List[str]] = None, insurance_providers: Optional[List[str]] = None, limit: int = 5, rating_weight: float = 0.7) -> List[dict]`**:
    *   **Purpose**: Searches a database for hospitals matching specified criteria such as location, type, and insurance.
    *   **Parameters**:
        *   `user_lat` (float): Latitude of the user's current location.
        *   `user_lon` (float): Longitude of the user's current location.
        *   `hospital_types` (Optional[List[str]]): A list of desired hospital types (e.g., "children's", "general").
        *   `insurance_providers` (Optional[List[str]]): A list of accepted insurance providers.
        *   `limit` (int): The maximum number of hospitals to return (default is 5).
        *   `rating_weight` (float): A weighting factor to balance rating vs distance in scoring.
    *   **Returns**: A list of dictionaries, where each dictionary represents a matching hospital with its details (e.g., name, distance).
    *   **Internal Logic**: Internally calls `tools.hospital_lookup.hospital_lookup_wrapper`.

## Settings and Configuration
### `settings/config.py` Module

This module centralizes all configuration parameters and constants used throughout the Voice Hospital Finder Bot project. It handles environment variable loading, sets up logging, and defines various constants for different modules and functionalities.

**Initialization & Setup:**

*   **Environment Variables**: Uses `dotenv.load_dotenv()` to load environment variables from a `.env` file, ensuring sensitive information and configurable settings are externalized.
*   **spaCy NLP Model**: Attempts to load the `en_core_web_sm` spaCy model for Natural Language Processing tasks. Provides an error message and instructions if the model is not found, making setup clearer.
    *   `NLP_MODEL`: Global variable holding the loaded spaCy model or `None` if loading fails.
*   **Logging**: Configures a basic logging system using Python's `logging` module, with the `uvicorn.logger` used for output.
    *   `LOGGER`: The configured logger instance.

**Configuration Variables:**

*   **`CITY_COORDINATES` (dict)**:
    *   Predefined dictionary mapping city names (mainly within the UAE) to their respective latitude and longitude coordinates. Used for geographical lookups and data generation.

*   **`HOSPITAL_TYPES` (list)**:
    *   A comprehensive list of various hospital specialties and types supported by the system for categorization and search filtering.

*   **`INSURANCE_PROVIDERS` (list)**:
    *   A list of recognized insurance providers that the system can filter hospitals by.

*   **Transcriber Config**:
    *   `TRANSCRIBER_OPENAI_MODEL` (str): Specifies the OpenAI Whisper model used for audio transcription (e.g., "whisper-1").
    *   `TRANSCRIBER_LANGUAGE` (str): Defines the language for transcription (e.g., "en" for English).

*   **Recognizer Config**:
    *   `FUZZY_MATCH_THRESHOLD` (int): Threshold for fuzzy matching in NLU, typically for spaCy entity recognition.
    *   `USE_LLM_FOR_RECOGNITION` (bool): Flag to enable or disable the use of a Large Language Model for entity recognition, alongside or instead of spaCy.
    *   `RECOGNIZER_MODEL` (str): Identifier for the LLM used for recognition if `USE_LLM_FOR_RECOGNITION` is true (e.g., "google/gemini-2.0-flash-001").
    *   `RECOGNIZER_TEMPERATURE` (float): Controls the randomness of the LLM's output during recognition.

*   **Hospital Data Generation Config**:
    *   `HOSPITAL_DB_FOLDER` (str): The directory where hospital data files are stored.
    *   `HOSPITAL_DB_FILE_NAME` (str): The name of the SQLite database file containing hospital information.
    *   `VECTOR_DB_FOLDER` (str): Sub-directory within `HOSPITAL_DB_FOLDER` for the FAISS vector database.

*   **Clarifier Config**:
    *   `CLARIFIER_MODEL` (str): Identifier for the LLM used by the clarifier component to generate clarification prompts.
    *   `CLARIFIER_TEMPERATURE` (float): Controls the randomness of the clarifier LLM's output.

*   **Text-to-Dialogue Config**:
    *   `TEXT_TO_DIALOGUE` (bool): Flag to enable or disable LLM-driven refinement of text into more conversational dialogue before TTS.
    *   `TEXT_TO_DIALOGUE_MODEL` (str): Identifier for the LLM used for text-to-dialogue conversion.
    *   `TEXT_TO_DIALOGUE_TEMPERATURE` (float): Controls the randomness of the text-to-dialogue LLM's output.

*   **`MAX_TURNS` (int)**:
    *   The maximum number of conversational turns allowed in a session before the bot attempts to conclude or exit.

*   **Hospital Finder Config**:
    *   `DEFAULT_N_HOSPITALS_TO_RETURN` (int): The default number of top hospitals to return in search results.
    *   `DEFAULT_DISTANCE_KM` (int): The default search radius in kilometers.
    *   `LOOKUP_MODE` (str): Specifies the lookup mechanism ("simple" for direct DB lookup, "rag" for RAG-augmented lookup).
    *   `RAG_GROUNDER_MODEL` (str): The LLM used for grounding RAG results when `LOOKUP_MODE` is "rag" and `GROUND_WITH_FINE_TUNE` is `False`.
    *   `RAG_GROUNDER_TEMPERATURE` (float): Temperature for the RAG grounder LLM.
    *   `GROUND_WITH_FINE_TUNE` (bool): Flag to enable/disable the use of the fine-tuned QLoRA model for RAG grounding.

*   **LLM Fine-Tuning Configurations (QLoRA)**:
    *   `BASE_MODEL` (str): Base Hugging Face model for fine-tuning.
    *   `TOKENIZER_MODEL` (str): Tokenizer model for fine-tuning.
    *   `FINE_TUNE_OUTPUT_DIR` (str): Directory where the fine-tuned model and tokenizer are saved.
    *   `FINE_TUNE_DATA_PATH` (str): Path to the fine-tuning dataset (e.g., "data/insurance_data.json").
    *   `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `MAX_SEQ_LEN`, `GRADIENT_ACCUMULATION_STEPS`, `FP16`: Training hyperparameters for QLoRA.
    *   `SAVE_STEPS`, `LOGGING_STEPS`: Checkpoint and logging frequency.
    *   `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `TARGET_MODULES`, `LORA_TASK_TYPE`: QLoRA specific parameters.
    *   `USE_SAFETENSORS`, `LOAD_IN_4BIT`, `OPTIMIZER`: Optional/advanced configuration for model loading and optimization.

### `settings/prompts.py` Module

This module stores various system and user prompts used to guide Large Language Models (LLMs) for specific tasks such as natural language understanding (NLU), text-to-dialogue conversion, and RAG grounding. These prompts are crucial for instructing LLMs on expected output formats, constraints, and contextual information.

**Prompt Definitions:**

*   **`RECOGNIZER_SYSTEM_PROMPT` (str)**:
    *   **Purpose**: Instructs an LLM to act as an expert AI assistant for extracting structured information (location, hospital type, insurance) from user queries about hospitals. It defines the rules for extraction, normalization, and handling of missing information.
    *   **Key Directives**:
        *   Output must be valid JSON with specific keys (`"location"`, `"hospital_type"`, `"insurance"`).
        *   Values must be lowercase.
        *   Normalization of locations and hospital types.
        *   Strict adherence to the JSON schema.

*   **`RECOGNIZER_USER_PROMPT` (str)**:
    *   **Purpose**: Provides the actual user query to the LLM, prompting it to apply the rules defined in `RECOGNIZER_SYSTEM_PROMPT`.
    *   **Placeholder**: Includes `{query_text}` which is dynamically replaced with the user's transcribed query.

*   **`TEXT_TO_DIALOGUE_SYSTEM_PROMPT` (str)**:
    *   **Purpose**: Guides an LLM to rephrase or enrich input text into a conversational, polite, and dialogue-friendly sentence suitable for text-to-speech. It emphasizes natural language, contextual relevance, and specific output formatting.
    *   **Key Directives**:
        *   Output must be valid JSON with `"dialogue"` and optional `"tone"` fields.
        *   Preserve original meaning, but simplify for spoken delivery.
        *   Avoid robotic phrasing and technical jargon unsuitable for speech.
        *   `Explicitly` prohibits mentioning distances or proximity unless stated by the user.

*   **`TEXT_TO_DIALOGUE_USER_PROMPT` (str)**:
    *   **Purpose**: Feeds the text to be converted into dialogue to the LLM, instructing it to follow the guidelines from `TEXT_TO_DIALOGUE_SYSTEM_PROMPT`.
    *   **Placeholder**: Includes `{text}` for the input text to be processed.

*   **`RAG_GROUNDER_SYSTEM_PROMPT` (str)**:
    *   **Purpose**: Instructs an LLM to act as a hospital information assistant, answering user queries based *only* on provided hospital context. It emphasizes accurate and concise responses, especially for insurance and specialty queries.
    *   **Key Directives**:
        *   Strictly use the provided `hospital_context` for answers.
        *   If information isn't in context, state that.
        *   Focus on direct answers, listing hospitals or confirming information.
        *   Mention supported insurance providers.
        *   Output must be valid JSON with `"hospital_ids"` (list of integers) and `"dialogue"` (string).

*   **`RAG_GROUNDER_USER_PROMPT` (str)**:
    *   **Purpose**: Provides the user's query and the retrieved hospital context to the RAG grounder LLM.
    *   **Placeholders**: Includes `{user_query}`, `{user_loc}`, `{hospital_context}`, and other relevant user input details for context.

## Tools
### `tools/hospital_lookup.py` Module

This module provides functionality for looking up hospitals based on geographical location, hospital types, and insurance providers. It includes distance calculation and a scoring mechanism to rank relevant hospitals.

**Functions:**

*   **`def haversine_distance(lat1, lon1, lat2, lon2) -> float`**:
    *   **Purpose**: Calculates the great-circle distance between two points on the Earth (specified in decimal degrees) using the Haversine formula.
    *   **Parameters**:
        *   `lat1` (float): Latitude of the first point.
        *   `lon1` (float): Longitude of the first point.
        *   `lat2` (float): Latitude of the second point.
        *   `lon2` (float): Longitude of the second point.
    *   **Returns**: The distance between the two points in kilometers.

*   **`async def find_hospitals_async(user_lat: float, user_lon: float, hospital_types: Optional[List[str]] = None, insurance_providers: Optional[List[str]] = None, limit: int = 5, rating_weight: float = 0.7) -> List[dict]`**:
    *   **Purpose**: Asynchronously retrieves and filters hospitals from the database based on the provided criteria.
    *   **Parameters**:
        *   `user_lat` (float): Latitude of the user's location.
        *   `user_lon` (float): Longitude of the user's location.
        *   `hospital_types` (Optional[List[str]]): A list of required hospital types.
        *   `insurance_providers` (Optional[List[str]]): A list of accepted insurance providers.
        *   `limit` (int): The maximum number of hospitals to return.
        *   `rating_weight` (float): The weighting factor for balancing rating vs distance in scoring.
    *   **Process**:
        1.  Fetches all hospitals from the database.
        2.  Converts hospital data into a Pandas DataFrame for efficient processing.
        3.  Filters hospitals based on `hospital_types` and `insurance_providers`.
        4.  Calculates the `haversine_distance` for each hospital from the user's location.
        5.  Normalizes ratings and distances to create a composite "score".
        6.  Sorts hospitals by this score and returns the top `limit` results.
    *   **Returns**: A list of dictionaries, each representing a hospital with its calculated distance and score.

*   **`async def hospital_lookup_wrapper(user_lat: float, user_lon: float, hospital_types: Optional[List[str]] = None, insurance_providers: Optional[List[str]] = None, limit: int = 5, rating_weight: float = 0.7) -> List[dict]`**:
    *   **Purpose**: A wrapper function that logs the lookup request and calls `find_hospitals_async` to perform the actual search.
    *   **Parameters**: (Same as `find_hospitals_async`).
    *   **Returns**: A list of dictionaries, each representing a matching hospital.

### `tools/recognize.py` Module

This module provides the core functionality for Natural Language Understanding (NLU) by extracting key entities like `location`, `hospital_type`, and `insurance` from user queries. It supports both spaCy-based fuzzy matching and LLM-based extraction.

**Class:**

*   **`QueryRecognizer`**:
    *   **Purpose**: Orchestrates the entity extraction process.
    *   **`__init__(self, hospital_types: Optional[list] = None, insurance_providers: Optional[list] = None)`**:
        *   **Purpose**: Initializes the `QueryRecognizer` instance, loading the spaCy NLP model and setting up predefined lists of hospital types and insurance providers.
        *   **Parameters**:
            *   `hospital_types` (Optional[list]): A list of known hospital types for fuzzy matching. Defaults to `config.HOSPITAL_TYPES`.
            *   `insurance_providers` (Optional[list]): A list of known insurance providers for fuzzy matching. Defaults to `config.INSURANCE_PROVIDERS`.
        *   **Raises**: `ValueError` if `NLP_MODEL` fails to load.
    *   **`_extract_location(self, doc) -> Optional[str]`**:
        *   **Purpose**: Extracts geographic or facility-based location entities using spaCy's Named Entity Recognition (NER).
        *   **Parameters**:
            *   `doc`: A spaCy `Doc` object representing the parsed user query.
        *   **Returns**: The extracted location as a string or `None` if no relevant entity is found.
    *   **`_extract_hospital_types(self, text: str) -> Optional[list]`**:
        *   **Purpose**: Detects hospital specialties in the query text using fuzzy string matching against a predefined list. It can handle abbreviations and partial words.
        *   **Parameters**:
            *   `text` (str): The raw query text.
        *   **Returns**: A list of matched hospital types (strings) or an empty list.
    *   **`_extract_insurance_providers(self, text: str) -> Optional[list]`**:
        *   **Purpose**: Detects insurance providers in the query text using fuzzy string matching. It can also identify generic mentions of "insurance."
        *   **Parameters**:
            *   `text` (str): The raw query text.
        *   **Returns**: A list of matched insurance providers (strings) or an empty list.
    *   **`async def _extract_with_llm(self, query_text: str) -> LLMResponseModel`**:
        *   **Purpose**: Uses an LLM to extract structured entities based on defined system and user prompts.
        *   **Parameters**:
            *   `query_text` (str): The user's query text.
        *   **Returns**: An `LLMResponseModel` object containing the extracted intent, location, hospital type, and insurance.
        *   **Internal Logic**: Formats prompts with the query text and calls `async_llm_client.beta.chat.completions.parse` for LLM interaction. Ensures all extracted string values are converted to lowercase.
*   **`async def recognize(self, query_text: str, uid: str, use_llm: bool = False) -> Dict`**:
    *   **Purpose**: The main entry point for entity recognition. It dynamically chooses between LLM-based or spaCy-based extraction.
    *   **Parameters**:
        *   `query_text` (str): The transcribed user query.
        *   `uid` (str): Unique identifier for the session.
        *   `use_llm` (bool): If `True`, uses the LLM for extraction; otherwise, defaults to spaCy + fuzzy matching.
    *   **Raises**: `ValueError` if `query_text` is empty.
    *   **Returns**: A dictionary containing `uid`, `query`, `intent`, `location`, `location_coordinates` (latitude, longitude tuple), `hospital_type` (list), `insurance` (list), `n_hospitals` (int), and `distance_km` (int or float).
    *   **Internal Logic**: Calls `_extract_with_llm` if `use_llm` is `True`, otherwise uses internal spaCy and fuzzy matching methods. Retrieves `location_coordinates` asynchronously.

**Functions:**

*   **`async def recognize_wrapper(query_text: str, uid: Optional[str] = None, use_llm: bool = False) -> Dict`**:
    *   **Purpose**: A convenience wrapper function to initialize `QueryRecognizer` and call its `recognize` method.
    *   **Parameters**:
        *   `query_text` (str): The transcribed user query.
        *   `uid` (str, optional): A unique ID for the session. If `None`, a new one is generated.
        *   `use_llm` (bool, default `False`): Flag to determine whether to use LLM for extraction.
    *   **Returns**: A dictionary with extracted entities and metadata, similar to the `recognize` method, including `n_hospitals` and `distance_km`.

### `tools/rag_retrieve.py` Module

This module implements the core logic for Retriever-Augmented Generation (RAG), combining semantic search from a FAISS vector database with LLM-based grounding to generate contextually relevant responses about hospitals. It also supports optional integration of a fine-tuned QLoRA model for specialized grounding, particularly for insurance-related queries.

**Classes:**

*   **`HospitalRAGRetriever`**:
    *   **Purpose**: Manages the retrieval of relevant hospital documents from the vector database and orchestrates the grounding of these results into natural language responses.
    *   **`__init__(self, vector_db_path_override: str = None)`**:
        *   **Purpose**: Initializes the retriever, loads the FAISS vector database, and optionally loads the fine-tuned QLoRA model.
        *   **Parameters**:
            *   `vector_db_path_override` (Optional[str]): Path to the vector database, defaults to `config.vector_db_path`.
        *   **Process**:
            1.  Initializes `OpenAIEmbeddings` as the embedding model.
            2.  Calls `_load_vector_db()` to load the FAISS index.
            3.  If `config.GROUND_WITH_FINE_TUNE` is `True`, calls `_load_finetuned_model()` to load the QLoRA model and tokenizer.
    *   **`_load_vector_db(self)`**:
        *   **Purpose**: Loads the FAISS vector database from the specified path. Raises `FileNotFoundError` if the database does not exist.
    *   **`_load_finetuned_model(self)`**:
        *   **Purpose**: Loads the base LLM and then wraps it with the PEFT (Parameter-Efficient Fine-tuning) adapter to enable QLoRA inference. The model is moved to the appropriate device (CUDA or CPU) and set to evaluation mode.
    *   **`_haversine_distance(lat1, lon1, lat2, lon2) -> float`**: (Static Method)
        *   **Purpose**: Calculates the Haversine distance between two sets of coordinates.
    *   **`_build_query(user_input: Dict) -> str`**: (Static Method)
        *   **Purpose**: Constructs a comprehensive query string from various user input parameters for semantic search in the vector database.
    *   **`retrieve(self, user_input: Dict, extra_results: int = 2) -> List[Dict]`**:
        *   **Purpose**: Performs a similarity search in the FAISS vector database based on the `user_input` query.
        *   **Parameters**:
            *   `user_input` (Dict): Contains user's query details (location, hospital types, insurance, etc.).
            *   `extra_results` (int): Number of extra results to retrieve beyond `n_hospitals` to ensure sufficient context for grounding.
        *   **Process**:
            1.  Constructs a `query_text` using `_build_query`.
            2.  Performs `vector_db.similarity_search` to get top relevant `Document` objects.
            3.  Filters and sorts retrieved documents based on user's location, `intent` (`find_nearest`, `find_best`), and `distance_km_radius`.
        *   **Returns**: A list of dictionaries representing the filtered and sorted hospital metadata.
    *   **`async def ground_with_insurance_info_qlora(self, user_query: str, hospitals_context: List[dict]) -> str`**:
        *   **Purpose**: Uses the fine-tuned QLoRA LLM to generate a natural language response grounded in the provided `hospitals_context` for a given `user_query`, specifically tailored for insurance-related queries.
        *   **Process**:
            1.  Formats an instruction-response prompt for the QLoRA model, including the `user_query` and structured `hospitals_context`.
            2.  Tokenizes the prompt and generates a response using the loaded QLoRA model.
            3.  Decodes the generated token IDs into text and extracts the relevant response.
        *   **Returns**: A natural language string generated by the QLoRA model.
    *   **`async def ground_results(self, user_input: dict, retrieved_hospitals: List[dict]) -> RAGGroundedResponseModel`**:
        *   **Purpose**: Orchestrates the grounding of retrieved hospital information into a user-friendly dialogue. It conditionally uses either the fine-tuned QLoRA model or a standard LLM.
        *   **Process**:
            1.  Checks if `config.GROUND_WITH_FINE_TUNE` is enabled and if the retrieved context is relevant for QLoRA grounding.
            2.  If so, calls `ground_with_insurance_info_qlora`.
            3.  Otherwise, formats system and user prompts (`settings/prompts.py`) with the `user_input` and `retrieved_hospitals` context.
            4.  Sends the prompts to a standard LLM (`async_llm_client`) for response generation.
            5.  Parses the LLM's response into an `RAGGroundedResponseModel`.
        *   **Returns**: An `RAGGroundedResponseModel` containing `hospital_ids` and the generated `dialogue`.

**Functions:**

*   **`async def rag_search_wrapper(...) -> Tuple[List[dict], str]`**:
    *   **Purpose**: A convenience wrapper to perform a complete RAG search given user parameters. It initializes the retriever, performs retrieval, and grounds the results.
    *   **Parameters**: Accepts various user input parameters (e.g., `user_query`, `user_loc`, `hospital_types`, `insurance_providers`), and an initialized `HospitalRAGRetriever` instance.
    *   **Process**:
        1.  Constructs a `user_input` dictionary.
        2.  Calls `retriever.retrieve` to get relevant hospitals.
        3.  Calls `retriever.ground_results` to generate a natural language dialogue based on the retrieved hospitals.
        4.  Maps the grounded hospital IDs back to their full details.
    *   **Returns**: A tuple containing a list of selected hospital dictionaries and the generated dialogue string.

### `tools/record.py` Module

This module provides functionality for recording audio input from the user. It leverages `asyncio` for non-blocking I/O operations, ensuring the recording process integrates smoothly within an asynchronous application flow.

**Functions:**

*   **`async def record_audio_wrapper(uid: str, duration: int = 5) -> dict`**:
    *   **Purpose**: Asynchronously records user voice input and saves it to a WAV file.
    *   **Parameters**:
        *   `uid` (str): A unique session identifier, used to name the audio file for tracking.
        *   `duration` (int, optional): The duration of the recording in seconds. Defaults to 5 seconds.
    *   **Process**:
        1.  Ensures the `audios/input` directory exists.
        2.  Constructs a unique audio file path using the provided `uid`.
        3.  Logs the start of the recording.
        4.  Executes the `record_audio` utility function (a blocking I/O operation) in a separate thread using `asyncio.to_thread` to prevent blocking the event loop.
        5.  Logs the completion of the recording.
    *   **Returns**: A dictionary containing the `uid` and the `input_audio_path` to the saved audio file.
    *   **Raises**: Catches and logs any exceptions that occur during recording, then re-raises them.

### `tools/text_to_speech.py` Module

This module manages the conversion of text into spoken audio using OpenAI's Text-to-Speech (TTS) API. It also includes an optional feature for refining input text into a more conversational dialogue format using an LLM before TTS conversion.

**Functions:**

*   **`async def text_to_speech_wrapper(text: str, uid: str, output_dir: str = "audios/output", convert_to_dialogue: bool = False) -> dict`**:
    *   **Purpose**: Converts an input text string into an audio file (MP3 format) using the OpenAI TTS API. Optionally, it can enhance the text into a more natural dialogue using an LLM.
    *   **Parameters**:
        *   `text` (str): The original text to be converted to speech.
        *   `uid` (str): A unique identifier for the session, used for naming the output audio file.
        *   `output_dir` (str, optional): The directory where the generated audio file will be saved. Defaults to "audios/output".
        *   `convert_to_dialogue` (bool, optional): If `True`, the text will first be processed by an LLM to make it more dialogue-friendly. Defaults to `False`.
    *   **Process**:
        1.  Ensures the `output_dir` exists.
        2.  Generates a unique filename for the audio using the `uid` and a UUID.
        3.  If `convert_to_dialogue` is `True`, it calls `text_to_dialogue` to refine the text.
        4.  Uses `asyncio.to_thread` to call OpenAI's `client.audio.speech.create` in a thread-safe asynchronous manner to perform the TTS conversion.
        5.  Saves the streamed audio response to the specified `output_dir`.
    *   **Returns**: A dictionary containing the `uid`, original `text`, the potentially modified `dialogue`, `tone`, the `audio_path` of the generated speech file, and a boolean indicating if an `llm_used` for dialogue conversion.
    *   **Raises**: Catches and logs any exceptions during the TTS process.

*   **`async def text_to_dialogue(text: str) -> TTSResponseModel`**:
    *   **Purpose**: Refines a given text into a more natural and conversational dialogue format using an LLM, based on predefined system and user prompts.
    *   **Parameters**:
        *   `text` (str): The input text to be converted into dialogue.
    *   **Process**:
        1.  Formates the `TEXT_TO_DIALOGUE_USER_PROMPT` with the input `text`.
        2.  Sends the system and user prompts to an LLM via `async_llm_client.beta.chat.completions.parse`.
        3.  The LLM's response is parsed into a `TTSResponseModel`.
        4.  If the LLM does not return a dialogue, the original text is used as a fallback.
    *   **Returns**: A `TTSResponseModel` object containing the `dialogue` and `tone`.

### `tools/transcribe.py` Module

This module provides functionality for transcribing audio files into text using the OpenAI Whisper API. It is designed to work asynchronously and handle both WAV and MP3 audio formats, making it suitable for converting spoken queries into a textual format for further processing by NLU or LLM components.

**Functions:**

*   **`async def transcribe_wrapper(audio_path: str, uid: str) -> dict`**:
    *   **Purpose**: Transcribes an audio file using OpenAI's Whisper API.
    *   **Parameters**:
        *   `audio_path` (str): The file path to the audio to be transcribed.
        *   `uid` (str): A unique session identifier for tracking and logging.
    *   **Process**:
        1.  Checks if the specified `audio_path` exists, raising `FileNotFoundError` if not.
        2.  Logs the start of the transcription process.
        3.  Opens the audio file in binary read mode (`"rb"`).
        4.  Uses `asyncio.to_thread` to safely call the blocking `client.audio.transcriptions.create` method of the OpenAI client, which sends the audio file to the Whisper API for transcription.
        5.  Extracts the `transcribed_text` from the API response.
        6.  Logs the completed transcription.
    *   **Returns**: A dictionary containing the `uid`, the `audio_path`, and the `transcribed_text`.
    *   **Raises**: `FileNotFoundError` if the audio file doesn't exist, and catches/logs other `Exception`s during the transcription process, then re-raises them.

## Utilities
### `utils/utils.py` Module

This module contains various utility functions that support the core functionalities of the Voice Hospital Finder Bot, including geolocation services, audio recording and playback, and state management.

**Functions:**

*   **`def get_lat_long(location_name) -> tuple`**:
    *   **Purpose**: Retrieves the latitude and longitude coordinates for a given geographical location name.
    *   **Parameters**:
        *   `location_name` (str): The name of the location (e.g., "Dubai", "Al Ain").
    *   **Process**: Uses the `geopy.geocoders.Nominatim` service to look up the coordinates.
    *   **Returns**: A tuple `(latitude, longitude)` if the location is found; otherwise, `None`. Logs debug information or errors.

*   **`async def record_audio(output_filename="input_audio.wav", duration=5, rate=44100, chunk=1024, channels=1)`**:
    *   **Purpose**: Records audio from the system's microphone and saves it as a WAV file.
    *   **Parameters**:
        *   `output_filename` (str): The desired path and filename for the recorded WAV file. Defaults to "input_audio.wav".
        *   `duration` (int): The length of the recording in seconds. Defaults to 5.
        *   `rate` (int): The audio sample rate (samples per second). Defaults to 44100 Hz.
        *   `chunk` (int): The number of audio frames per buffer. Defaults to 1024.
        *   `channels` (int): The number of audio channels (e.g., 1 for mono). Defaults to 1.
    *   **Process**:
        1.  Initializes `pyaudio`.
        2.  Opens an audio stream with specified parameters.
        3.  Records audio frames for the given `duration`.
        4.  Closes the audio stream and terminates `pyaudio`.
        5.  Ensures the output directory exists.
        6.  Saves the recorded frames to a WAV file at `output_filename`.
    *   **Returns**: The `output_filename` upon successful recording.

*   **`async def play_audio(audio_path, chunk=1024)`**:
    *   **Purpose**: Plays a specified WAV audio file through the system's speakers.
    *   **Parameters**:
        *   `audio_path` (str): The file path to the WAV audio file to be played.
        *   `chunk` (int): The number of frames per buffer for playback. Defaults to 1024.
    *   **Process**:
        1.  Checks if the `audio_path` exists. Logs an error and returns if not.
        2.  Opens the WAV file.
        3.  Initializes `pyaudio` and opens an audio output stream.
        4.  Reads and plays audio data in chunks until the file ends.
        5.  Stops the stream, closes it, and terminates `pyaudio`.
    *   **Outputs**: Plays audio; logs debug information.

*   **`async def summarize_conversation(final_state: HospitalFinderState)`**:
    *   **Purpose**: Prints a summary of the entire conversational session, including initial queries, clarifications, hospitals found, and the bot's final response.
    *   **Parameters**:
        *   `final_state` (`HospitalFinderState`): The final state object of the conversation.
    *   **Outputs**: Prints formatted summary to the console.

*   **`async def save_state(state: HospitalFinderState, output_dir="outputs") -> str`**:
    *   **Purpose**: Saves the current `HospitalFinderState` object to a JSON file.
    *   **Parameters**:
        *   `state` (`HospitalFinderState`): The state object to be saved.
        *   `output_dir` (str): The directory where the JSON file will be saved. Defaults to "outputs".
    *   **Process**:
        1.  Ensures the `output_dir` exists.
        2.  Converts the `HospitalFinderState` Pydantic model to a dictionary.
        3.  Constructs a unique file path using the state's `uid`.
        4.  Asynchronously writes the dictionary to a JSON file using `asyncio.to_thread`.
    *   **Returns**: The file path to the saved JSON state file.
