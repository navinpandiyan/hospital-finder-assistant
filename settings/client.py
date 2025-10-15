import os
from openai import OpenAI, AsyncOpenAI

# Initialize LLM Client
llm_client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )

async_llm_client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )


openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
)
