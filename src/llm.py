"""IBM watsonx.ai LLM client wrapper with retries, timeouts, and graceful degradation."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import (
    GUARDIAN_MODEL_ID,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    LLM_TIMEOUT_SECONDS,
    MAX_NEW_TOKENS_BATCH_SCORING_BASE,
    MAX_NEW_TOKENS_BATCH_SCORING_PER_Q,
    MAX_NEW_TOKENS_GUARDIAN,
    MAX_NEW_TOKENS_QUESTIONS,
    MAX_NEW_TOKENS_REPORT,
    MAX_NEW_TOKENS_SCORING,
    SCORING_TEMPERATURE,
    GENERATION_TEMPERATURE,
    WATSONX_API_KEY,
    WATSONX_MODEL_ID,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""


class LLMQuotaError(LLMError):
    """Raised when quota / rate limit is exceeded."""


class LLMAuthError(LLMError):
    """Raised on authentication failures."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM call exceeds the timeout."""


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from LLM output."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> str:
    """Try to extract the first JSON object or array from *text*."""
    text = _strip_markdown_fences(text)
    # Find first { or [
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start == -1:
        return text
    # Find matching closing bracket using a simple depth counter
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


class WatsonxLLMClient:
    """Thin wrapper around ibm-watsonx-ai for text generation."""

    def __init__(self) -> None:
        self._client: Any = None
        self._model: Any = None
        self._guardian_model: Any = None
        self._available: bool = False
        self._init_client()

    def _init_client(self) -> None:
        """Initialise the watsonx.ai client; set _available=False on any error."""
        try:
            from ibm_watsonx_ai import Credentials, APIClient
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            credentials = Credentials(
                url=WATSONX_URL,
                api_key=WATSONX_API_KEY,
            )
            self._client = APIClient(credentials)
            self._GenParams = GenParams

            self._model = ModelInference(
                model_id=WATSONX_MODEL_ID,
                credentials=credentials,
                project_id=WATSONX_PROJECT_ID,
            )

            if GUARDIAN_MODEL_ID:
                self._guardian_model = ModelInference(
                    model_id=GUARDIAN_MODEL_ID,
                    credentials=credentials,
                    project_id=WATSONX_PROJECT_ID,
                )

            self._available = True
            logger.info("WatsonxLLMClient initialised with model %s", WATSONX_MODEL_ID)
        except Exception as exc:
            logger.error("Failed to initialise WatsonxLLMClient: %s", type(exc).__name__)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Core generation method
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = GENERATION_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS_QUESTIONS,
    ) -> str:
        """Call the LLM and return the generated text."""
        if not self._available:
            raise LLMError("LLM client is not available. Check your credentials.")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        try:
            response = self._generate_with_retry(full_prompt, params)
            return response
        except RetryError as exc:
            raise LLMError(f"LLM call failed after {LLM_MAX_RETRIES} retries") from exc

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        stop=stop_after_attempt(LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=LLM_RETRY_BASE_DELAY, min=1, max=30),
        reraise=False,
    )
    def _generate_with_retry(self, prompt: str, params: dict[str, Any]) -> str:
        """Inner method that tenacity retries on transient errors."""
        import signal
        import threading

        def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise TimeoutError(f"LLM call timed out after {LLM_TIMEOUT_SECONDS}s")

        # SIGALRM only works on Unix and only in the main thread
        in_main_thread = threading.current_thread() is threading.main_thread()
        sigalrm_set = False
        try:
            if in_main_thread:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(LLM_TIMEOUT_SECONDS)
                sigalrm_set = True
        except (AttributeError, OSError):
            pass  # Windows or unavailable

        try:
            response = self._model.generate_text(
                prompt=prompt,
                params=params,
            )
            return response or ""
        except Exception as exc:
            exc_name = type(exc).__name__
            exc_str = str(exc)
            if "401" in exc_str or "unauthorized" in exc_str.lower():
                raise LLMAuthError("Authentication failed. Check WATSONX_API_KEY.") from exc
            if "429" in exc_str or "quota" in exc_str.lower() or "rate limit" in exc_str.lower():
                raise LLMQuotaError("Rate limit or quota exceeded on watsonx.ai.") from exc
            if "timeout" in exc_str.lower() or isinstance(exc, TimeoutError):
                raise TimeoutError(exc_str) from exc
            raise ConnectionError(f"watsonx.ai call failed: {exc_name}") from exc
        finally:
            try:
                if sigalrm_set:
                    signal.alarm(0)
            except (AttributeError, OSError):
                pass

    # ------------------------------------------------------------------
    # JSON generation helper (with repair loop)
    # ------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = GENERATION_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS_QUESTIONS,
        max_repair_attempts: int = 2,
    ) -> dict[str, Any] | list[Any]:
        """Generate and parse JSON, with up to *max_repair_attempts* repair retries."""
        from src.prompts import REPAIR_PROMPT

        raw = self.generate(prompt, system_prompt, temperature, max_new_tokens)
        extracted = _extract_json(raw)

        for attempt in range(max_repair_attempts + 1):
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                if attempt < max_repair_attempts:
                    logger.warning("JSON parse failed (attempt %d), sending repair prompt", attempt + 1)
                    repair = REPAIR_PROMPT.format(previous_response=extracted[:2000])
                    raw = self.generate(repair, "", 0.0, max_new_tokens)
                    extracted = _extract_json(raw)
                else:
                    raise LLMError(
                        "LLM returned invalid JSON after repair attempts. "
                        f"Raw output (truncated): {extracted[:300]}"
                    )
        # unreachable
        return {}

    # ------------------------------------------------------------------
    # Guardian check
    # ------------------------------------------------------------------

    def guardian_check(self, content: str) -> dict[str, Any]:
        """
        Optional Granite Guardian safety check.
        Returns a dict with 'safe' bool.
        Falls back to {'safe': True} if guardian is unavailable.
        """
        if not self._guardian_model:
            return {"safe": True, "source": "guardian_unavailable"}

        from src.prompts import GUARDIAN_PROMPT

        try:
            prompt = GUARDIAN_PROMPT.format(content=content[:2000])
            raw = self._guardian_model.generate_text(
                prompt=prompt,
                params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS_GUARDIAN},
            )
            extracted = _extract_json(raw or "")
            result = json.loads(extracted)
            result["source"] = "guardian_model"
            return result
        except Exception as exc:
            logger.warning("Guardian check failed (%s), falling back to rule-based", type(exc).__name__)
            return {"safe": True, "source": "guardian_fallback"}

    # ------------------------------------------------------------------
    # Scoring convenience
    # ------------------------------------------------------------------

    def generate_scoring(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> dict[str, Any] | list[Any]:
        return self.generate_json(
            prompt,
            system_prompt,
            temperature=SCORING_TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS_SCORING,
        )

    def generate_batch_scoring(
        self,
        prompt: str,
        system_prompt: str = "",
        n_questions: int = 1,
    ) -> dict[str, Any] | list[Any]:
        """Batch scoring with token budget scaled to number of questions."""
        max_tokens = MAX_NEW_TOKENS_BATCH_SCORING_BASE + n_questions * MAX_NEW_TOKENS_BATCH_SCORING_PER_Q
        return self.generate_json(
            prompt,
            system_prompt,
            temperature=SCORING_TEMPERATURE,
            max_new_tokens=max_tokens,
        )

    def generate_report(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> dict[str, Any] | list[Any]:
        return self.generate_json(
            prompt,
            system_prompt,
            temperature=SCORING_TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS_REPORT,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_client_instance: WatsonxLLMClient | None = None


def get_llm_client() -> WatsonxLLMClient:
    """Return the module-level singleton LLM client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = WatsonxLLMClient()
    return _client_instance
