"""LLM provider factory."""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_llm_client(llm_config: dict) -> Any:
    """Create an LLM client from config."""
    provider = llm_config["provider"]
    api_key = os.getenv(llm_config["api_key_env"])

    if not api_key:
        raise ValueError(
            f"Missing API key in environment variable: {llm_config['api_key_env']}"
        )

    if provider == "openai_compatible":
        return OpenAI(
            api_key=api_key,
            base_url=llm_config["base_url"],
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")