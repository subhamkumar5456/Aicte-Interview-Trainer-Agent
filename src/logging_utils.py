"""Safe logging utilities with PII redaction."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from src.config import LOG_DIR, REPORT_LOG_FILE


# ---------------------------------------------------------------------------
# PII redaction patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I), "[EMAIL]"),
    # International phone numbers (various formats)
    (
        re.compile(
            r"(?:\+?\d[\s\-\.]?){7,14}\d",
        ),
        "[PHONE]",
    ),
    # Aadhaar-style 12-digit number (India national ID)
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[AADHAAR]"),
    # PAN-style ID (India): 5 letters, 4 digits, 1 letter
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN]"),
    # Credit-card-like 16-digit numbers
    (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "[CC_NUM]"),
    # US SSN
    (re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b"), "[SSN]"),
    # API keys / bearer tokens (rough heuristic: long alphanumeric strings)
    (
        re.compile(r"(?:api[_\-]?key|bearer|token|secret|password)\s*[:=]\s*\S+", re.I),
        "[REDACTED_CREDENTIAL]",
    ),
]


def redact_pii(text: str) -> str:
    """Apply all PII redaction patterns to *text* and return the result."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


class RedactingFormatter(logging.Formatter):
    """Logging formatter that redacts PII from log records."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return redact_pii(original)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with PII-redacting formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        _ensure_log_dir()
        handler = logging.StreamHandler()
        handler.setFormatter(
            RedactingFormatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Structured report logging (for "Report a bad response" feature)
# ---------------------------------------------------------------------------

def log_bad_response_report(
    session_id: str,
    question_id: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a redacted bad-response report to the JSONL log file."""
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question_id": question_id,
        "reason": redact_pii(reason),
        "extra": {k: redact_pii(str(v)) for k, v in (extra or {}).items()},
    }
    with open(REPORT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Security event logging
# ---------------------------------------------------------------------------

def log_security_event(event_type: str, details: str, session_id: str = "") -> None:
    """Log a security event (injection attempt, guardrail trigger) without raw content."""
    logger = get_logger("security")
    # Never log the raw content — only the type and a sanitized detail
    safe_details = redact_pii(details[:200])  # cap length
    logger.warning("SECURITY_EVENT type=%s session=%s details=%s", event_type, session_id, safe_details)
