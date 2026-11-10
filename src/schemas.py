"""Pydantic v2 schemas for all structured LLM outputs."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Interview question
# ---------------------------------------------------------------------------

class InterviewQuestion(BaseModel):
    """A single generated interview question with metadata."""

    question_id: str = Field(..., description="Unique ID for this question")
    question_text: str = Field(..., min_length=10, max_length=1000)
    role: str = Field(..., description="Target role")
    interview_type: Literal["technical", "behavioral", "mixed"]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    grounded_in_source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source questions from the bank used to generate this",
    )
    resume_context_hint: str = Field(
        default="",
        description="Which part of the resume this question relates to",
    )
    is_general_fallback: bool = Field(
        default=False,
        description="True when retrieval returned nothing relevant",
    )

    @field_validator("question_text")
    @classmethod
    def no_illegal_questions(cls, v: str) -> str:
        # Use whole-word matching to avoid false positives inside words like
        # "manage", "package", "language", "nationality" containing "age", etc.
        illegal_topics = [
            "marital status", "married", "religion", "religious",
            "pregnant", "pregnancy", "caste", "disability", "age",
            "nationality", "race",
        ]
        lower = v.lower()
        for topic in illegal_topics:
            pattern = r"\b" + re.escape(topic) + r"\b"
            if re.search(pattern, lower):
                raise ValueError(
                    f"Generated question contains a potentially illegal topic: '{topic}'"
                )
        return v


class GeneratedQuestions(BaseModel):
    """Wrapper returned by the question-generation LLM call."""

    questions: list[InterviewQuestion] = Field(..., min_length=1, max_length=7)

    @field_validator("questions")
    @classmethod
    def at_least_one(cls, v: list[InterviewQuestion]) -> list[InterviewQuestion]:
        if not v:
            raise ValueError("questions list must not be empty")
        return v


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------

class AnswerScore(BaseModel):
    """Structured scoring result for a single answer."""

    question_id: str
    score: int = Field(..., ge=0, le=10, description="Score from 0 to 10")
    feedback: str = Field(..., min_length=10, max_length=2000)
    model_answer: str = Field(..., min_length=10, max_length=3000)
    covered_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: int | float | str) -> int:
        """Clamp score to 0-10 regardless of model output."""
        try:
            score = int(float(str(v)))
        except (ValueError, TypeError):
            score = 0
        return max(0, min(10, score))

    @field_validator("feedback", "model_answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank")
        return v.strip()


# ---------------------------------------------------------------------------
# Batch scoring schemas (new batch flow)
# ---------------------------------------------------------------------------

class BatchAnswerItem(BaseModel):
    """One item in the batch scoring request — question + candidate answer."""

    question_id: str
    question_text: str
    key_points: list[str] = Field(default_factory=list)
    answer: str  # already validated / sanitized before this point


class BatchScoringResult(BaseModel):
    """The LLM's scoring result for a single Q&A pair inside a batch response."""

    question_id: str
    score: int = Field(..., ge=0, le=10)
    feedback: str = Field(..., min_length=10, max_length=2000)
    model_answer: str = Field(..., min_length=10, max_length=3000)
    covered_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: int | float | str) -> int:
        try:
            score = int(float(str(v)))
        except (ValueError, TypeError):
            score = 0
        return max(0, min(10, score))

    @field_validator("feedback", "model_answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank")
        return v.strip()

    def to_answer_score(self) -> "AnswerScore":
        return AnswerScore(
            question_id=self.question_id,
            score=self.score,
            feedback=self.feedback,
            model_answer=self.model_answer,
            covered_points=self.covered_points,
            missed_points=self.missed_points,
        )


class BatchScoringResponse(BaseModel):
    """Full JSON structure returned by the batch scoring LLM call."""

    results: list[BatchScoringResult] = Field(..., min_length=1)
    strengths: list[str] = Field(default_factory=list, max_length=10)
    improvements: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_result_ids(self) -> "BatchScoringResponse":
        """Ensure no duplicate question_ids in results."""
        seen: set[str] = set()
        deduped: list[BatchScoringResult] = []
        for r in self.results:
            if r.question_id not in seen:
                seen.add(r.question_id)
                deduped.append(r)
        self.results = deduped
        return self


# ---------------------------------------------------------------------------
# Final report (built in Python from batch scoring response)
# ---------------------------------------------------------------------------

class FinalReport(BaseModel):
    """Aggregated report at the end of the interview session."""

    average_score: float = Field(..., description="Average score, clamped to 0-10 by model validator")
    per_question_scores: list[dict[str, object]] = Field(
        ..., description="[{question_id, question_text, score}]"
    )
    top_strengths: list[str] = Field(..., min_length=1, max_length=5)
    top_improvements: list[str] = Field(..., min_length=1, max_length=5)
    summary: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def check_score_bounds(self) -> "FinalReport":
        self.average_score = max(0.0, min(10.0, self.average_score))
        return self


# ---------------------------------------------------------------------------
# Guardian / safety check
# ---------------------------------------------------------------------------

class GuardianResult(BaseModel):
    """Result from the optional Granite Guardian call."""

    safe: bool
    risk_category: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: str = ""
