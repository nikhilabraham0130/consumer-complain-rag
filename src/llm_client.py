"""
Resilient LLM Client supporting DeepSeek-V3, Google Gemini, and OpenAI.

All three providers support the universal OpenAI SDK format:
- DeepSeek: base_url="https://api.deepseek.com", model="deepseek-chat"
- Gemini (Free Tier): base_url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.5-flash"
- OpenAI: base_url="https://api.openai.com/v1", model="gpt-4o-mini"
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Type, TypeVar
import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("llm_client")

T = TypeVar("T", bound=BaseModel)


class LLMClientError(Exception):
    """Base exception for LLM client communication and parsing errors."""
    pass


class LLMConfigurationError(LLMClientError):
    """Raised when API key or model configuration is missing or invalid."""
    pass


class LLMClient:
    """
    Unified client supporting DeepSeek, Gemini, and OpenAI.
    Automatically detects which provider key is present in .env.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = settings.LLM_MAX_RETRIES,
        timeout: int = settings.LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout

        # Resolve provider: DeepSeek -> Gemini -> OpenAI
        raw_deepseek = api_key or settings.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")
        raw_gemini = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        raw_openai = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

        # Clean string helper
        def _clean(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            s = val.strip().strip("'\"")
            return s if s and not s.startswith("your_") else None

        deepseek_key = _clean(raw_deepseek)
        gemini_key = _clean(raw_gemini)
        openai_key = _clean(raw_openai)

        if deepseek_key:
            self.provider = "DeepSeek"
            self.api_key = deepseek_key
            self.base_url = base_url or settings.DEEPSEEK_BASE_URL
            self.model = model or settings.DEEPSEEK_MODEL
        elif gemini_key:
            self.provider = "Google Gemini"
            self.api_key = gemini_key
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.model = model or "gemini-2.5-flash"
        elif openai_key:
            self.provider = "OpenAI"
            self.api_key = openai_key
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or "gpt-4o-mini"
        else:
            raise LLMConfigurationError(
                "No valid LLM API key found. Please add DEEPSEEK_API_KEY or GEMINI_API_KEY to your .env file."
            )

        logger.info(f"Initialized LLM Client using provider: [bold cyan]{self.provider}[/bold cyan] (model={self.model})")

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = settings.LLM_TEMPERATURE,
        json_mode: bool = False,
    ) -> str:
        """
        Executes a chat completion with exponential backoff on transient network/rate errors.
        """
        response_format = {"type": "json_object"} if json_mode else None
        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise LLMClientError(f"Received empty response from {self.provider} API.")
                return content.strip()

            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
                if attempt == self.max_retries:
                    logger.error(f"{self.provider} API failed after {self.max_retries} attempts: {e}")
                    raise LLMClientError(f"{self.provider} API call failed: {e}") from e
                logger.warning(
                    f"Transient API error ({type(e).__name__}). Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})..."
                )
                time.sleep(delay)
                delay *= 2.0
            except openai.APIError as e:
                logger.error(f"Non-retryable {self.provider} API error: {e}")
                raise LLMClientError(f"Fatal API error: {e}") from e

        raise LLMClientError("Max retries exceeded without a response.")

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        temperature: float = 0.0,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        schema_instruction = (
            f"\n\nCRITICAL OUTPUT INSTRUCTION:\n"
            f"You MUST return a JSON object strictly matching this schema definition:\n"
            f"{schema_json}\n"
            f"Ensure all required fields are present with exact matching keys."
        )

        augmented_messages = [dict(m) for m in messages]
        if augmented_messages and augmented_messages[0]["role"] == "system":
            augmented_messages[0] = {
                "role": "system",
                "content": augmented_messages[0]["content"] + schema_instruction,
            }
        else:
            augmented_messages.insert(0, {"role": "system", "content": schema_instruction})

        raw_json_str = self.generate_chat(
            messages=augmented_messages,
            temperature=temperature,
            json_mode=True,
        )

        try:
            parsed_data = json.loads(raw_json_str)
            return schema.model_validate(parsed_data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to validate LLM output against schema {schema.__name__}: {e}\nRaw output: {raw_json_str}")
            raise LLMClientError(f"Structured output validation failed: {e}") from e
