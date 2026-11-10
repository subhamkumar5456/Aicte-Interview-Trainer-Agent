"""
Tests for src/schemas.py — pydantic model validation.

All tests run fully offline (no LLM calls).
"""
from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("WATSONX_API_KEY", "test_key")
os.environ.setdefault("WATSONX_PROJECT_ID", "test_project")

import pydantic

from src.schemas import (
    AnswerScore,
    FinalReport,
    GeneratedQuestions,
    GuardianResult,
    InterviewQuestion,
)


# ---------------------------------------------------------------------------
# InterviewQuestion
# ---------------------------------------------------------------------------

class TestInterviewQuestion:
    def test_valid_question(self):
        q = InterviewQuestion(
            question_id="q1",
            question_text="Describe how you would design a RESTful API.",
            role="Full-Stack Developer",
            interview_type="technical",
        )
        assert q.question_id == "q1"
        assert q.difficulty == "medium"  # default
        assert not q.is_general_fallback  # default

    def test_question_too_short_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            InterviewQuestion(
                question_id="q1",
                question_text="Why?",  # too short
                role="Full-Stack Developer",
                interview_type="technical",
            )

    def test_invalid_interview_type(self):
        with pytest.raises(pydantic.ValidationError):
            InterviewQuestion(
                question_id="q1",
                question_text="Describe the lifecycle of a React component.",
                role="Full-Stack Developer",
                interview_type="essay",  # invalid
            )

    def test_invalid_difficulty(self):
        with pytest.raises(pydantic.ValidationError):
            InterviewQuestion(
                question_id="q1",
                question_text="Explain event-driven architecture in Node.js.",
                role="Full-Stack Developer",
                interview_type="technical",
                difficulty="extreme",  # invalid
            )

    @pytest.mark.parametrize(
        "question_text",
        [
            "Are you married and does your marital status affect availability?",
            "What is your religion and how does it affect your schedule?",
            "Are you pregnant or planning to have children?",
            "What is your caste background?",
            "Do you have any disability that would affect your work?",
        ],
    )
    def test_illegal_questions_rejected(self, question_text: str):
        with pytest.raises(pydantic.ValidationError, match="illegal"):
            InterviewQuestion(
                question_id="q1",
                question_text=question_text,
                role="Full-Stack Developer",
                interview_type="behavioral",
            )

    def test_general_fallback_flag(self):
        q = InterviewQuestion(
            question_id="q1",
            question_text="Describe your approach to writing clean and maintainable code.",
            role="Full-Stack Developer",
            interview_type="technical",
            is_general_fallback=True,
        )
        assert q.is_general_fallback

    def test_grounded_in_source_ids_default_empty(self):
        q = InterviewQuestion(
            question_id="q1",
            question_text="How do you handle merge conflicts in a team environment?",
            role="Full-Stack Developer",
            interview_type="behavioral",
        )
        assert q.grounded_in_source_ids == []


# ---------------------------------------------------------------------------
# GeneratedQuestions
# ---------------------------------------------------------------------------

class TestGeneratedQuestions:
    def test_valid_list(self):
        data = {
            "questions": [
                {
                    "question_id": "q1",
                    "question_text": "Walk me through how you would design a caching strategy.",
                    "role": "Full-Stack Developer",
                    "interview_type": "technical",
                    "difficulty": "hard",
                    "grounded_in_source_ids": [],
                    "resume_context_hint": "",
                    "is_general_fallback": False,
                }
            ]
        }
        gq = GeneratedQuestions.model_validate(data)
        assert len(gq.questions) == 1

    def test_empty_list_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            GeneratedQuestions.model_validate({"questions": []})

    def test_too_many_questions_rejected(self):
        questions = []
        for i in range(8):  # max is 7
            questions.append(
                {
                    "question_id": f"q{i}",
                    "question_text": f"Explain concept number {i} in software engineering.",
                    "role": "Full-Stack Developer",
                    "interview_type": "technical",
                }
            )
        with pytest.raises(pydantic.ValidationError):
            GeneratedQuestions.model_validate({"questions": questions})


# ---------------------------------------------------------------------------
# AnswerScore
# ---------------------------------------------------------------------------

class TestAnswerScore:
    def test_valid_score(self):
        data = {
            "question_id": "q1",
            "score": 8,
            "feedback": "Excellent answer covering all the key points comprehensively.",
            "model_answer": "An ideal answer would also mention horizontal scaling.",
            "covered_points": ["load balancing", "caching"],
            "missed_points": ["horizontal scaling"],
        }
        score = AnswerScore.model_validate(data)
        assert score.score == 8
        assert len(score.covered_points) == 2

    def test_score_float_coerced(self):
        data = {
            "question_id": "q1",
            "score": 6.7,
            "feedback": "Good attempt with room for improvement.",
            "model_answer": "Cover A, B, C more thoroughly.",
            "covered_points": ["A"],
            "missed_points": ["B", "C"],
        }
        score = AnswerScore.model_validate(data)
        assert score.score == 6  # truncated to int

    def test_score_string_coerced(self):
        data = {
            "question_id": "q1",
            "score": "9",
            "feedback": "Excellent comprehensive answer.",
            "model_answer": "Very close to ideal; consider adding X.",
            "covered_points": ["main point"],
            "missed_points": [],
        }
        score = AnswerScore.model_validate(data)
        assert score.score == 9

    def test_score_above_10_clamped(self):
        data = {
            "question_id": "q1",
            "score": 100,
            "feedback": "Perfect answer.",
            "model_answer": "Just as given.",
            "covered_points": [],
            "missed_points": [],
        }
        score = AnswerScore.model_validate(data)
        assert score.score == 10

    def test_score_negative_clamped(self):
        data = {
            "question_id": "q1",
            "score": -3,
            "feedback": "Very poor answer with no relevant content.",
            "model_answer": "You should have covered A, B, and C.",
            "covered_points": [],
            "missed_points": ["A", "B", "C"],
        }
        score = AnswerScore.model_validate(data)
        assert score.score == 0

    def test_empty_feedback_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            AnswerScore(
                question_id="q1",
                score=5,
                feedback="",
                model_answer="A great answer covers X and Y.",
            )

    def test_empty_model_answer_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            AnswerScore(
                question_id="q1",
                score=5,
                feedback="Reasonable attempt but missed some key points.",
                model_answer="",
            )

    def test_whitespace_only_feedback_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            AnswerScore(
                question_id="q1",
                score=5,
                feedback="   \t  ",
                model_answer="An ideal answer would cover A and B.",
            )


# ---------------------------------------------------------------------------
# FinalReport
# ---------------------------------------------------------------------------

class TestFinalReport:
    def test_valid_report(self):
        data = {
            "average_score": 7.5,
            "per_question_scores": [
                {"question_id": "q1", "question_text": "Describe REST.", "score": 8},
                {"question_id": "q2", "question_text": "Explain CI/CD.", "score": 7},
            ],
            "top_strengths": ["Clear communication", "Good depth on REST"],
            "top_improvements": ["Add more examples", "Cover edge cases"],
            "summary": "Solid performance across most questions.",
        }
        report = FinalReport.model_validate(data)
        assert report.average_score == 7.5
        assert len(report.top_strengths) == 2

    def test_average_score_clamped(self):
        data = {
            "average_score": 15.0,
            "per_question_scores": [],
            "top_strengths": ["Good"],
            "top_improvements": ["Improve"],
            "summary": "",
        }
        report = FinalReport.model_validate(data)
        assert report.average_score == 10.0

    def test_no_strengths_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            FinalReport(
                average_score=7.0,
                per_question_scores=[],
                top_strengths=[],  # min_length=1
                top_improvements=["Improve A"],
            )


# ---------------------------------------------------------------------------
# GuardianResult
# ---------------------------------------------------------------------------

class TestGuardianResult:
    def test_safe_result(self):
        r = GuardianResult(safe=True, risk_category="none", confidence=0.99)
        assert r.safe

    def test_unsafe_result(self):
        r = GuardianResult(safe=False, risk_category="hate", confidence=0.95)
        assert not r.safe
        assert r.risk_category == "hate"

    def test_confidence_out_of_range(self):
        with pytest.raises(pydantic.ValidationError):
            GuardianResult(safe=True, confidence=1.5)  # max 1.0
