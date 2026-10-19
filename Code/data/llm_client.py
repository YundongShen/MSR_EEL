"""Thin LLM client wrapper (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper over the OpenAI client (or a mock for offline testing).

    Parameters
    ----------
    provider:
        ``"openai"`` (default) or ``"mock"`` for unit tests.
    model:
        Model name, e.g. ``"gpt-4o"`` or ``"Qwen/Qwen2.5-Coder-32B-Instruct"``.
    api_base:
        Override base URL (for local vLLM servers).
    max_tokens:
        Maximum tokens in the completion.
    temperature:
        Sampling temperature.
    max_retries:
        Number of retry attempts on transient errors.
    retry_delay:
        Seconds to wait between retries.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_base: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_base = api_base
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None or self._provider == "mock":
            return
        import os
        if self._provider == "anthropic":
            try:
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError("Install anthropic: pip install anthropic") from exc
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
            return
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai>=1.0") from exc
        kwargs: dict = {}
        if self._api_base:
            kwargs["base_url"] = self._api_base
            # Local vLLM server doesn't need a real key, but the OpenAI client requires one.
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._client = OpenAI(**kwargs)

    def complete(self, system: str, user: str) -> str | None:
        """Return the model's response, or ``None`` after all retries fail."""
        self._ensure_client()
        if self._provider == "mock":
            return self._mock_response(user)

        if self._provider == "anthropic":
            for attempt in range(self._max_retries):
                try:
                    response = self._client.messages.create(  # type: ignore[union-attr]
                        model=self._model,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return response.content[0].text or ""
                except Exception as exc:
                    log.warning("Anthropic call failed (attempt %d/%d): %s", attempt + 1, self._max_retries, exc)
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_delay * (attempt + 1))
            return None

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, self._max_retries, exc)
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
        return None

    @staticmethod
    def _mock_response(user_prompt: str) -> str:
        """Deterministic stub for offline testing."""
        return (
            "--- a/billing/discounts.py\n"
            "+++ b/billing/discounts.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     if user.tier == 'premium':\n"
            "-        return price * 0.10\n"
            "+        return price * 0.20\n"
        )
