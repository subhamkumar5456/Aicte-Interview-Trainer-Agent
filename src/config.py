"""Configuration loading and constants for Interview Trainer."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------

def _require(name: str) -> str:
    """Return env var value or exit with a clear message."""
    val = os.getenv(name, "").strip()
    if not val:
        sys.exit(
            f"[interview-trainer] Missing required environment variable: {name}\n"
            "Copy .env.example to .env and fill in your credentials."
        )
    return val


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Watsonx credentials
WATSONX_API_KEY: str = _require("WATSONX_API_KEY")
WATSONX_PROJECT_ID: str = _require("WATSONX_PROJECT_ID")
WATSONX_URL: str = _optional("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID: str = _optional(
    "WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct"
)
GUARDIAN_MODEL_ID: str = _optional("GUARDIAN_MODEL_ID", "")  # optional

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
GENERATION_TEMPERATURE: float = 0.5
SCORING_TEMPERATURE: float = 0.2
MAX_NEW_TOKENS_QUESTIONS: int = 1024
MAX_NEW_TOKENS_SCORING: int = 1024
MAX_NEW_TOKENS_REPORT: int = 512
MAX_NEW_TOKENS_GUARDIAN: int = 128
# Batch scoring: scale with N (N * 250 + 300, max 7 questions → 2050)
MAX_NEW_TOKENS_BATCH_SCORING_BASE: int = 300   # fixed overhead
MAX_NEW_TOKENS_BATCH_SCORING_PER_Q: int = 250  # per answered question

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Batch flow: 2 LLM calls per session (1 generation + 1 scoring)
MAX_LLM_CALLS_PER_SESSION: int = 2
MIN_SECONDS_BETWEEN_CALLS: float = 2.0
DAILY_LLM_CALL_CAP: int = 500  # across all sessions

# ---------------------------------------------------------------------------
# Input limits
# ---------------------------------------------------------------------------
RESUME_MAX_CHARS: int = 6000
RESUME_MIN_CHARS: int = 50
ANSWER_MAX_CHARS: int = 2000
ANSWER_MIN_CHARS: int = 10
FILE_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB
ALLOWED_FILE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"application/pdf", "text/plain"})

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
RAG_TOP_K: int = 8
QUESTIONS_JSON_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "data", "questions.json"
)

# ---------------------------------------------------------------------------
# LLM call limits
# ---------------------------------------------------------------------------
LLM_TIMEOUT_SECONDS: int = 45
LLM_MAX_RETRIES: int = 3
LLM_RETRY_BASE_DELAY: float = 1.5  # seconds, exponential base

# ---------------------------------------------------------------------------
# Interview config
# ---------------------------------------------------------------------------
AVAILABLE_ROLES: list[str] = [
    "Full-Stack Developer",
    "AI/ML Engineer",
    "Cybersecurity Analyst",
]
AVAILABLE_INTERVIEW_TYPES: list[str] = ["technical", "behavioral", "mixed"]
MIN_QUESTIONS: int = 3
MAX_QUESTIONS: int = 7

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "..", "logs")
REPORT_LOG_FILE: str = os.path.join(LOG_DIR, "report_log.jsonl")
