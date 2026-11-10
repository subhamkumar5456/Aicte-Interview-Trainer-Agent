"""
All guardrail logic for Interview Trainer — implements all 10 guardrail categories.

Each public function returns a GuardrailResult dataclass.
No business logic lives here; this module is purely defensive.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from src.config import (
    ALLOWED_FILE_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    ANSWER_MAX_CHARS,
    ANSWER_MIN_CHARS,
    AVAILABLE_INTERVIEW_TYPES,
    AVAILABLE_ROLES,
    FILE_MAX_BYTES,
    MAX_LLM_CALLS_PER_SESSION,
    MIN_SECONDS_BETWEEN_CALLS,
    RESUME_MAX_CHARS,
    RESUME_MIN_CHARS,
)
from src.logging_utils import get_logger, log_security_event, redact_pii

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Outcome of a guardrail check."""

    passed: bool
    message: str = ""
    is_self_harm: bool = False
    modified_text: str | None = None  # sanitized version of input, if applicable


# ---------------------------------------------------------------------------
# 1. Input guardrails — length, file, character cleaning
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXCESSIVE_WHITESPACE_RE = re.compile(r"[ \t]{4,}")
_NULL_BYTE_RE = re.compile(r"\x00")


def sanitize_text(text: str) -> str:
    """
    Strip control characters, null bytes, normalize unicode,
    and collapse excessive whitespace. Return the cleaned string.
    """
    text = _NULL_BYTE_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Collapse runs of spaces/tabs (but preserve newlines for readability)
    text = _EXCESSIVE_WHITESPACE_RE.sub(" ", text)
    return text.strip()


def validate_resume(text: str) -> GuardrailResult:
    """Validate resume text length and content."""
    text = sanitize_text(text)
    if len(text) < RESUME_MIN_CHARS:
        return GuardrailResult(
            passed=False,
            message=f"Resume is too short (minimum {RESUME_MIN_CHARS} characters). Please paste your full resume.",
        )
    if len(text) > RESUME_MAX_CHARS:
        return GuardrailResult(
            passed=False,
            message=(
                f"Resume exceeds the {RESUME_MAX_CHARS:,}-character limit. "
                "Please trim it or paste only the most relevant sections."
            ),
        )
    if _is_binary_garbage(text):
        return GuardrailResult(passed=False, message="Resume appears to contain binary data. Please paste plain text.")
    return GuardrailResult(passed=True, modified_text=text)


def validate_answer(text: str) -> GuardrailResult:
    """Validate an interview answer."""
    text = sanitize_text(text)
    if len(text) < ANSWER_MIN_CHARS:
        return GuardrailResult(
            passed=False,
            message=f"Answer is too short (minimum {ANSWER_MIN_CHARS} characters).",
        )
    if len(text) > ANSWER_MAX_CHARS:
        # Truncate with notice rather than hard block
        truncated = text[:ANSWER_MAX_CHARS]
        return GuardrailResult(
            passed=True,
            message=f"Answer was truncated to {ANSWER_MAX_CHARS:,} characters.",
            modified_text=truncated,
        )
    return GuardrailResult(passed=True, modified_text=text)


def _is_binary_garbage(text: str) -> bool:
    """Heuristic: flag text with >10% non-printable, non-whitespace characters."""
    if not text:
        return False
    non_print = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
    return (non_print / len(text)) > 0.10


def validate_file_upload(filename: str, content: bytes, mime_type: str = "") -> GuardrailResult:
    """
    Validate an uploaded file: size, extension, MIME type.
    Returns GuardrailResult with modified_text=extracted text on success.
    """
    if len(content) > FILE_MAX_BYTES:
        return GuardrailResult(
            passed=False,
            message=f"File exceeds the 2 MB size limit ({len(content) / 1024 / 1024:.1f} MB uploaded).",
        )

    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_FILE_EXTENSIONS:
        return GuardrailResult(
            passed=False,
            message=f"File type '{ext}' is not allowed. Please upload a .pdf or .txt file.",
        )

    if mime_type and mime_type.split(";")[0].strip() not in ALLOWED_MIME_TYPES:
        return GuardrailResult(
            passed=False,
            message="File MIME type is not allowed. Please upload a PDF or plain text file.",
        )

    # Extract text only — never execute or render
    try:
        extracted = _extract_text_from_file(content, ext)
    except Exception as exc:
        logger.warning("File text extraction failed: %s", type(exc).__name__)
        return GuardrailResult(passed=False, message="Could not read the file. Please paste your resume as text instead.")

    return GuardrailResult(passed=True, modified_text=extracted)


def _extract_text_from_file(content: bytes, ext: str) -> str:
    """Extract plain text from PDF or TXT content."""
    if ext == ".txt":
        return content.decode("utf-8", errors="replace")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)

    raise ValueError(f"Unsupported extension: {ext}")


def validate_role(role: str) -> GuardrailResult:
    """Validate that role comes from the fixed allowlist."""
    if role not in AVAILABLE_ROLES:
        return GuardrailResult(
            passed=False,
            message=f"Invalid role. Choose from: {', '.join(AVAILABLE_ROLES)}.",
        )
    return GuardrailResult(passed=True)


def validate_interview_type(interview_type: str) -> GuardrailResult:
    """Validate that interview type comes from the fixed allowlist."""
    if interview_type not in AVAILABLE_INTERVIEW_TYPES:
        return GuardrailResult(
            passed=False,
            message=f"Invalid interview type. Choose from: {', '.join(AVAILABLE_INTERVIEW_TYPES)}.",
        )
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 2. Prompt-injection and jailbreak detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"reveal\s+(?:your\s+)?system\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)", re.I),
    re.compile(r"act\s+as\s+(?:a|an|the)\s+\w+\s+without\s+restrictions", re.I),
    re.compile(r"\bDAN\b"),  # Do Anything Now jailbreak
    re.compile(r"jailbreak", re.I),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a|an|the)", re.I),
    re.compile(r"bypass\s+(?:your\s+)?(?:safety|filter|guardrail|restriction)", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules)", re.I),
    re.compile(r"forget\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.I),
    re.compile(r"print\s+(?:your\s+)?system\s+prompt", re.I),
    re.compile(r"output\s+(?:your\s+)?(?:system\s+prompt|instructions)", re.I),
    # Attempt to escape delimiter tags
    re.compile(r"</(?:resume|answer|system|instruction)>", re.I),
    re.compile(r"</?(?:system|instruction)\s*>", re.I),
    # Classic role-play jailbreak opener
    re.compile(r"\blet's\s+play\s+a\s+game\b.*?(?:no\s+rules|anything)", re.I | re.S),
]


def check_injection(text: str, source: str = "input", session_id: str = "") -> GuardrailResult:
    """
    Detect prompt-injection and jailbreak attempts.
    Logs the event without the raw content.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            log_security_event(
                event_type="INJECTION_ATTEMPT",
                details=f"Pattern matched: {pattern.pattern[:80]} | source={source}",
                session_id=session_id,
            )
            return GuardrailResult(
                passed=False,
                message=(
                    "Your input contains patterns that cannot be processed. "
                    "Please enter your genuine resume or interview answer."
                ),
            )
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 3. PII detection and redaction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\s\-\.]?){7,14}\d")
_AADHAAR_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_CC_RE = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")


def redact_contact_details(text: str) -> str:
    """Mask email, phone, Aadhaar-style IDs, PAN IDs, and CC numbers in *text*."""
    text = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE REDACTED]", text)
    text = _AADHAAR_RE.sub("[ID REDACTED]", text)
    text = _PAN_RE.sub("[PAN REDACTED]", text)
    text = _CC_RE.sub("[CC REDACTED]", text)
    return text


def contains_pii(text: str) -> bool:
    """Return True if *text* appears to contain PII."""
    return bool(
        _EMAIL_RE.search(text)
        or _PHONE_RE.search(text)
        or _AADHAAR_RE.search(text)
        or _PAN_RE.search(text)
    )


# ---------------------------------------------------------------------------
# 4. Content safety and scope
# ---------------------------------------------------------------------------

# Off-topic / scope enforcement
_OFF_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:diagnose|medical\s+advice|prescription|dosage|symptoms\s+of)\b", re.I),
    re.compile(r"\b(?:legal\s+advice|lawsuit|sue|court\s+order|attorney)\b", re.I),
    re.compile(r"\b(?:stock\s+tip|investment\s+advice|crypto\s+investment)\b", re.I),
    re.compile(r"\b(?:political\s+party|vote\s+for|election\s+campaign)\b", re.I),
    re.compile(r"\b(?:dating\s+advice|romantic\s+relationship\s+advice)\b", re.I),
]

# Harm blocklist (rule-based)
_HARM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:hate\s+speech|racial\s+slur|kill\s+all|genocide)\b", re.I), "hate"),
    (re.compile(r"\b(?:sexual\s+content|explicit\s+content|nsfw|pornograph)\b", re.I), "sexual"),
    (
        re.compile(
            r"\b(?:how\s+to\s+make\s+(?:a\s+)?bomb|synthesize\s+drugs|make\s+drugs|drug\s+synthesis)\b",
            re.I,
        ),
        "illegal_activity",
    ),
    (
        re.compile(
            r"\b(?:suicide\s+method|how\s+to\s+(?:kill|harm)\s+(?:myself|yourself)|self[\-\s]harm\s+instructions)\b",
            re.I,
        ),
        "self_harm",
    ),
    (re.compile(r"\b(?:child\s+abuse|child\s+exploitation|pedophil)\b", re.I), "child_safety"),
]

# Cybersecurity offensive content
_OFFENSIVE_CYBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:working\s+exploit|shellcode|buffer\s+overflow\s+payload)\b", re.I),
    re.compile(r"\b(?:ransomware\s+code|malware\s+source|keylogger\s+source)\b", re.I),
    re.compile(r"\b(?:sql\s+injection\s+payload\s+to\s+exfiltrate)\b", re.I),
    re.compile(r"\b(?:step[\-\s]by[\-\s]step\s+attack|penetrate\s+and\s+exfiltrate)\b", re.I),
]

_SELF_HARM_PATTERN = re.compile(
    r"\b(?:suicide|suicidal|kill\s+myself|end\s+my\s+life|self[\-\s]harm|cut\s+myself|hurt\s+myself)\b",
    re.I,
)

SELF_HARM_RESPONSE = (
    "It sounds like you might be going through a difficult time. "
    "Please consider speaking with a trusted friend, family member, or professional. "
    "If you are in crisis, please reach out to a helpline in your country — "
    "for example the 988 Suicide & Crisis Lifeline (US) or iCall (India). "
    "The interview trainer will be here when you're ready to continue."
)


def check_content_safety(text: str, session_id: str = "") -> GuardrailResult:
    """
    Check for harmful, off-topic, or restricted content.
    Returns GuardrailResult.is_self_harm=True for self-harm inputs.
    """
    # Self-harm check first — respond with support, not rejection
    if _SELF_HARM_PATTERN.search(text):
        log_security_event("SELF_HARM_SIGNAL", "Self-harm pattern detected", session_id)
        return GuardrailResult(
            passed=False,
            message=SELF_HARM_RESPONSE,
            is_self_harm=True,
        )

    for pattern, category in _HARM_PATTERNS:
        if pattern.search(text):
            log_security_event(f"HARMFUL_CONTENT_{category.upper()}", "Harm pattern matched", session_id)
            return GuardrailResult(
                passed=False,
                message="This content cannot be processed by the interview trainer. Please keep inputs relevant to your professional experience.",
            )

    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                message=(
                    "This request appears to be outside the scope of interview preparation. "
                    "I can only help you practise for job interviews."
                ),
            )

    for pattern in _OFFENSIVE_CYBER_PATTERNS:
        if pattern.search(text):
            log_security_event("OFFENSIVE_CYBER_CONTENT", "Offensive cyber pattern detected", session_id)
            return GuardrailResult(
                passed=False,
                message="Interview questions and answers must stay educational and defensive. Working exploits or malware content is not allowed.",
            )

    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 5. Output guardrails
# ---------------------------------------------------------------------------

_LEAKED_PROMPT_FRAGMENTS = [
    "STRICT RULES",
    "system prompt",
    "WATSONX_API_KEY",
    "WATSONX_PROJECT_ID",
    "FAIRNESS REMINDER",
    "SCORING RUBRIC",
]

_TOO_LONG_OUTPUT_CHARS = 5000


def check_output(text: str, context: str = "output") -> GuardrailResult:
    """
    Validate LLM output: check for leaked prompt text, excessive length,
    empty content, and re-run content safety.
    """
    if not text or not text.strip():
        return GuardrailResult(passed=False, message="The AI returned an empty response. Please try again.")

    if len(text) > _TOO_LONG_OUTPUT_CHARS:
        return GuardrailResult(
            passed=False,
            message="The AI response was unexpectedly long and was not shown for safety. Please try again.",
        )

    for fragment in _LEAKED_PROMPT_FRAGMENTS:
        if fragment.lower() in text.lower():
            log_security_event("PROMPT_LEAK_DETECTED", f"Fragment found in output: {fragment[:40]}", "")
            return GuardrailResult(
                passed=False,
                message="The response could not be shown due to a safety check. Please try again.",
            )

    # Re-run content safety on output
    safety = check_content_safety(text, session_id="output_check")
    if not safety.passed:
        return GuardrailResult(
            passed=False,
            message="The AI-generated response did not pass the content safety check and was not shown.",
        )

    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 6. Grounding and hallucination guardrails
# ---------------------------------------------------------------------------

def check_grounding(question_text: str, resume_text: str) -> GuardrailResult:
    """
    Warn if a generated question references a specific employer, project,
    certification, or skill that does not appear in the resume.
    This is a heuristic check — the LLM is also instructed not to hallucinate.
    """
    # Extract proper-noun-like tokens from the question (capitalized words ≥5 chars)
    proper_nouns = re.findall(r"\b[A-Z][a-z]{4,}\b", question_text)
    resume_lower = resume_text.lower()

    suspicious: list[str] = []
    for noun in proper_nouns:
        # Skip generic words
        if noun.lower() in {
            "describe", "explain", "discuss", "provide", "mention",
            "consider", "include", "example", "question", "answer",
        }:
            continue
        if noun.lower() not in resume_lower:
            suspicious.append(noun)

    if len(suspicious) > 3:
        return GuardrailResult(
            passed=False,
            message=(
                f"The generated question may reference details not found in the resume "
                f"({', '.join(suspicious[:3])}…). It was flagged for review."
            ),
        )
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 7. Fairness and bias guardrails
# ---------------------------------------------------------------------------

_BIAS_TERMS: list[str] = [
    "gender", "female", "male", "woman", "man", "girl", "boy",
    "age", "young", "old", "elder", "senior citizen",
    "religion", "muslim", "christian", "hindu", "sikh", "jewish",
    "caste", "brahmin", "dalit",
    "nationality", "foreigner", "immigrant",
    "disability", "disabled", "blind", "deaf", "wheelchair",
    "appearance", "attractive", "ugly", "beautiful", "fat", "thin",
    "college tier", "iit", "tier-1", "tier-2",
    "marital", "married", "single", "divorced",
    "pregnant", "pregnancy",
    "race", "black", "white", "asian", "hispanic",
]

_BIAS_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _BIAS_TERMS) + r")\b",
    re.I,
)


def check_bias_in_output(text: str) -> GuardrailResult:
    """
    Detect if LLM output mentions bias-sensitive attributes.
    Returns passed=False with a list of matched terms.
    """
    matches = _BIAS_RE.findall(text)
    unique_matches = list({m.lower() for m in matches})
    if unique_matches:
        return GuardrailResult(
            passed=False,
            message=f"Response mentions bias-sensitive terms ({', '.join(unique_matches[:5])}) and was filtered.",
        )
    return GuardrailResult(passed=True)


def filter_biased_sentences(text: str) -> str:
    """
    Remove sentences from *text* that contain bias-sensitive terms.
    Returns the filtered text.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean: list[str] = []
    for sentence in sentences:
        if not _BIAS_RE.search(sentence):
            clean.append(sentence)
    return " ".join(clean)


# ---------------------------------------------------------------------------
# 8. Rate limiter (per-session)
# ---------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """Simple in-process per-session rate limiter."""

    max_calls: int = MAX_LLM_CALLS_PER_SESSION
    min_interval: float = MIN_SECONDS_BETWEEN_CALLS
    _call_count: int = field(default=0, init=False)
    _last_call_time: float = field(default=0.0, init=False)

    def check(self) -> GuardrailResult:
        """Check if a new LLM call is allowed."""
        now = time.monotonic()

        if self._call_count >= self.max_calls:
            return GuardrailResult(
                passed=False,
                message=(
                    f"You have reached the session limit of {self.max_calls} AI requests. "
                    "Please start a new session to continue."
                ),
            )

        elapsed = now - self._last_call_time
        if elapsed < self.min_interval and self._last_call_time > 0:
            wait_time = round(self.min_interval - elapsed, 1)
            return GuardrailResult(
                passed=False,
                message=f"Please wait {wait_time}s before sending another request.",
            )

        return GuardrailResult(passed=True)

    def record_call(self) -> None:
        """Record that an LLM call was made."""
        self._call_count += 1
        self._last_call_time = time.monotonic()

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_calls - self._call_count)


# ---------------------------------------------------------------------------
# 9. Language check (heuristic)
# ---------------------------------------------------------------------------

def check_language(text: str) -> GuardrailResult:
    """
    Warn if input does not appear to be mostly English.
    Results may be less reliable for non-English input.
    """
    try:
        from langdetect import detect, LangDetectException  # type: ignore

        lang = detect(text[:500])
        if lang != "en":
            return GuardrailResult(
                passed=True,  # still allowed, but warn
                message=(
                    f"Your input appears to be in a language other than English (detected: {lang}). "
                    "Results may be less reliable."
                ),
            )
    except Exception:
        pass  # langdetect failure is non-fatal
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# 10. Secrets scan (output)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:api[_\-]?key|secret|password|token)\s*[:=]\s*\S{8,}", re.I),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),  # base64-like long strings
]


def scan_for_secrets(text: str) -> GuardrailResult:
    """Check that LLM output does not contain credential-like strings."""
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                passed=False,
                message="Response was blocked because it appeared to contain sensitive credential data.",
            )
    return GuardrailResult(passed=True)
