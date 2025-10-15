# Voice Hospital Finder Bot

## An AI-powered conversational agent to find nearby hospitals using voice commands.

![Voice Hospital Finder Bot](https://i.ibb.co/7d0JBJFZ/558128e7-157a-4c31-bb7b-c07414dc8cf5.png)

## Table of Contents
*   [Introduction](#introduction)
*   [Features](#features)
*   [Architecture Highlights](#architecture-highlights)
*   [Setup and Installation](#setup-and-installation)
*   [Usage](#usage)
*   [Configuration](#configuration)
*   [Detailed Documentation](#detailed-documentation)
*   [Contributing](#contributing)
*   [License](#license)

## Introduction
The Voice Hospital Finder Bot is an interactive application that allows users to search for hospitals using natural voice commands. Leveraging Speech-to-Text (STT), Natural Language Understanding (NLU), and Text-to-Speech (TTS) technologies, it provides a seamless and intuitive way to locate healthcare facilities based on criteria such as location, hospital type, and insurance providers.

## Features
*   **Voice-Activated Interface**: Interact effortlessly through natural spoken language.
*   **Intelligent Query Recognition**: Understands user intent, extracting location, hospital type, and insurance details from spoken queries.
*   **Accurate Hospital Search**: Finds and ranks hospitals by proximity and other specified criteria from an SQLite database.
*   **Conversational Flow**: Manages chat context to ask clarifying questions and guide users effectively.
*   **Flexible Configuration**: Easily adjust language models, search parameters, and conversational behavior.

## Architecture Highlights
The bot is built around a `langgraph` StateGraph, which dynamically manages the conversational flow. It integrates various AI models for voice processing and understanding, backed by custom tools for:
*   Transcribing audio input using OpenAI Whisper.
*   Recognizing entities (location, hospital type, insurance) from text using a configured Language Model.
*   Converting text responses back to speech using OpenAI TTS.
*   Performing detailed hospital lookups with distance calculations and intelligent scoring.

For an in-depth understanding of the architecture, state management, and core tools, please refer to the [Detailed Documentation](#detailed-documentation).

## Setup and Installation

### Prerequisites
*   Python 3.8+
*   `ffmpeg`: Essential for audio processing. Install via your system's package manager (e.g., `sudo apt-get install ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS, or download binaries for Windows).
*   spaCy English model:
    ```bash
    uv run -- spacy download en_core_web_sm
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

    **Alternatively, using `pip` and manual virtual environments:**
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

3.  **Set up Environment Variables**: Create a `.env` file in the root directory:
    ```
    OPENAI_API_KEY="your_openai_api_key_for_whisper_and_tts"
    LLM_API_KEY="your_llm_api_key_for_intent_identification" # For models configured in settings/config.py (e.g., Google Gemini)
    LLM_BASE_URL="your_llm_base_url" # Optional: e.g., for local LLMs or specific API endpoints
    ```
    *   `OPENAI_API_KEY` is required for Speech-to-Text (Whisper) and Text-to-Speech functions.
    *   `LLM_API_KEY` and `LLM_BASE_URL` are used for intent identification and query recognition if using models other than OpenAI, as configured in `settings/config.py`.

## Usage
To start the Voice Hospital Finder Bot, execute:
```bash
uv run app.py
# Or if using pip/venv: python app.py
```
The bot will verbally prompt you. Speak your queries clearly, such as:
*   "Find a cardiology hospital in Dubai."
*   "Show me hospitals that accept Aetna insurance in Abu Dhabi."
*   "Are there any pediatric hospitals near Al Ain?"

To end the conversation, use keywords like "stop", "exit", "quit", or "goodbye".

## Configuration
Key configurable parameters are located in `settings/config.py`. These include settings for:
*   NLP models (spaCy NER, various LLMs).
*   Hospital types and insurance providers data.
*   Speech-to-Text and Text-to-Speech models.
*   Conversational `MAX_TURNS`.
For a complete list and explanation, refer to the [Detailed Documentation](#detailed-documentation).

## Detailed Documentation
For a comprehensive guide, including in-depth architectural details, extensibility options, and advanced troubleshooting, please see [DOCUMENTATION.md](/docs/DOCUMENTATION.md).