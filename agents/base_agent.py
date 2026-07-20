import asyncio
import logging
import os
from abc import ABC, abstractmethod
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents. Every subclass must define its own Pydantic input/output schemas
    and implement the `run` method.
    """

    @abstractmethod
    async def run(self, input: BaseModel) -> BaseModel:
        """Execute the agent's core task."""
        pass

    @abstractmethod
    def get_fallback_output(self, input: BaseModel, error_msg: str = "Timeout occurred") -> BaseModel:
        """Return a default schema instance in case of failure or timeout."""
        pass

    async def run_with_timeout(self, input: BaseModel, timeout: float = 5.0) -> BaseModel:
        """Wraps run() in an asyncio timeout block. Gracefully returns fallback on timeout/crashes."""
        try:
            return await asyncio.wait_for(self.run(input), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("Agent %s timed out after %s seconds.", self.__class__.__name__, timeout)
            return self.get_fallback_output(input, "Timeout occurred")
        except Exception as e:
            log.error("Agent %s crashed: %s", self.__class__.__name__, e, exc_info=True)
            return self.get_fallback_output(input, f"Agent error: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def call_llm(self, system_prompt: str, user_prompt: str, response_format=None) -> str:
        """
        Unified LLM client caller with built-in tenacity retries.
        Falls back to dummy mock responses if neither API key is set.
        Supports GEMINI_API_KEY as a free fallback.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if not api_key and not gemini_key:
            log.warning("Neither OPENAI_API_KEY nor GEMINI_API_KEY is set in environment. Running in offline/mock mode.")
            return self._generate_mock_llm_response(system_prompt, user_prompt)

        base_url = None
        model = "gpt-4o-mini"

        # Prioritize Gemini if GEMINI_API_KEY is set and OPENAI_API_KEY is missing or the quota-exceeded key
        if gemini_key and (not api_key or "sk-proj-bxKY" in api_key):
            api_key = gemini_key
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            model = "gemini-1.5-flash"
            log.info("Using Gemini OpenAI-compatible API with gemini-1.5-flash")

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _generate_mock_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generates standard deterministic responses for offline dry-runs and testing."""
        prompt_lower = user_prompt.lower()
        system_lower = system_prompt.lower()

        # Fallacy agent mock (check first to avoid 'claiming' matching 'claim')
        if "fallac" in system_lower:
            if "everyone else is doing it" in prompt_lower or "populum" in prompt_lower:
                return '{"fallacy_type": "ad populum", "confidence": 0.92}'
            if "antichrist" in prompt_lower or "hominem" in prompt_lower:
                return '{"fallacy_type": "ad hominem", "confidence": 0.89}'
            return '{"fallacy_type": "no fallacy", "confidence": 0.95}'

        # Fact verification agent mock
        if "verdict" in system_lower or "verify" in system_lower:
            # Look at user prompt to customize verdict
            if "shingles" in prompt_lower:
                return '{"verdict": "True", "confidence": 0.95, "cited_chunks": ["who_shingles_0001"]}'
            if "abortion" in prompt_lower:
                return '{"verdict": "False", "confidence": 0.88, "cited_chunks": ["liar_abortion_03"]}'
            return '{"verdict": "Unverified", "confidence": 0.5, "cited_chunks": []}'

        # Claim extraction agent mock
        if "extract" in system_lower or "claim" in system_lower:
            return '{"claims": ["The inflation rate in Chicago increased by 10% in 2019.", "Nuclear power has caused millions of deaths."]}'

        return "{}"
