"""
Agent logic for Interview Trainer.

Orchestrates: RAG retrieval → question generation → batch answer scoring.
All LLM calls are wrapped with guardrails and pydantic validation.

Batch flow (2 LLM calls per session):
  Call 1 — generate_questions: ONE call returns all N questions at once.
  Call 2 — score_all_answers:  ONE call scores all submitted answers together
                                and returns the full report structure.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from src import rag
from src.guardrails import (
    GuardrailResult,
    RateLimiter,
    check_bias_in_output,
    check_content_safety,
    check_grounding,
    check_injection,
    check_output,
    filter_biased_sentences,
    redact_contact_details,
    scan_for_secrets,
    validate_answer,
)
from src.llm import LLMError, WatsonxLLMClient, get_llm_client
from src.logging_utils import get_logger
from src.prompts import (
    REPAIR_PROMPT,
    SYSTEM_PROMPT_BATCH_SCORING,
    SYSTEM_PROMPT_QUESTION_GEN,
    SYSTEM_PROMPT_REPORT,
    SYSTEM_PROMPT_SCORING,
    USER_PROMPT_BATCH_SCORING,
    USER_PROMPT_QUESTION_GEN,
    USER_PROMPT_REPORT,
    USER_PROMPT_SCORING,
)
from src.schemas import (
    AnswerScore,
    BatchAnswerItem,
    BatchScoringResponse,
    FinalReport,
    GeneratedQuestions,
    InterviewQuestion,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe JSON string rendering
# ---------------------------------------------------------------------------

def _safe_render(text: str) -> str:
    """Escape text for safe display — strip any HTML tags."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Question generation (LLM call 1)
# ---------------------------------------------------------------------------

def generate_questions(
    resume_text: str,
    role: str,
    interview_type: str,
    n_questions: int,
    redact_pii: bool,
    rate_limiter: RateLimiter,
    session_id: str = "",
) -> tuple[list[InterviewQuestion], list[dict[str, Any]], str]:
    """
    Generate ALL interview questions in a single LLM call using RAG + Granite.

    Returns:
        (questions, retrieved_sources, warning_message)
        - questions: validated list of InterviewQuestion
        - retrieved_sources: raw retrieved docs for "Why this question?" UI
        - warning_message: non-fatal warning to show in UI (empty if none)
    """
    warning = ""

    # Rate limit
    rate_check = rate_limiter.check()
    if not rate_check.passed:
        raise LLMError(rate_check.message)

    # Injection check on resume
    injection = check_injection(resume_text, source="resume", session_id=session_id)
    if not injection.passed:
        raise ValueError(injection.message)

    # Content safety check on resume
    safety = check_content_safety(resume_text, session_id=session_id)
    if not safety.passed:
        raise ValueError(safety.message)

    # Optionally redact PII before sending to LLM
    text_for_llm = redact_contact_details(resume_text) if redact_pii else resume_text

    # RAG retrieval
    retrieved = rag.retrieve_questions(
        query=f"{role} {interview_type} interview",
        role=role,
        interview_type=interview_type,
    )
    is_fallback = len(retrieved) == 0

    if is_fallback:
        warning = (
            "No closely matching questions found in the question bank. "
            "Generating general role-based questions."
        )

    retrieved_text = rag.format_retrieved_for_prompt(retrieved)

    # Build prompt
    system = SYSTEM_PROMPT_QUESTION_GEN.format(role=role)
    user = USER_PROMPT_QUESTION_GEN.format(
        n_questions=n_questions,
        interview_type=interview_type,
        role=role,
        retrieved_questions=retrieved_text,
        resume_text=text_for_llm,
    )

    # LLM call 1
    client: WatsonxLLMClient = get_llm_client()
    rate_limiter.record_call()

    raw_data = client.generate_json(prompt=user, system_prompt=system, temperature=0.5)

    # Pydantic validation
    try:
        generated = GeneratedQuestions.model_validate(raw_data)
    except Exception as exc:
        raise LLMError(f"Question generation returned an invalid schema: {exc}") from exc

    # Post-process: output guardrails on each question
    safe_questions: list[InterviewQuestion] = []
    for q in generated.questions:
        # Bias check
        bias = check_bias_in_output(q.question_text)
        if not bias.passed:
            logger.warning("Biased question filtered: %s", bias.message)
            continue

        # Output content safety
        out_check = check_output(q.question_text, context="question")
        if not out_check.passed:
            logger.warning("Question failed output check: %s", out_check.message)
            continue

        # Grounding check
        grounding = check_grounding(q.question_text, resume_text)
        if not grounding.passed:
            logger.warning("Grounding warning: %s", grounding.message)
            if not warning:
                warning = grounding.message

        if is_fallback:
            q = q.model_copy(update={"is_general_fallback": True})

        # Assign a unique ID if missing or duplicate
        if not q.question_id or q.question_id in {qq.question_id for qq in safe_questions}:
            q = q.model_copy(update={"question_id": f"q_{uuid.uuid4().hex[:6]}"})

        safe_questions.append(q)

    if not safe_questions:
        raise LLMError("All generated questions were filtered by safety checks. Please try again.")

    return safe_questions, retrieved, warning


# ---------------------------------------------------------------------------
# Batch answer scoring (LLM call 2)
# ---------------------------------------------------------------------------

def score_all_answers(
    questions: list[InterviewQuestion],
    answers: dict[str, str],          # question_id → raw answer text
    retrieved_sources: list[dict[str, Any]],
    role: str,
    rate_limiter: RateLimiter,
    session_id: str = "",
) -> tuple[dict[str, AnswerScore], dict[str, str], FinalReport, str]:
    """
    Score all candidate answers in a SINGLE LLM call.

    Input guardrails run on each answer locally before the batch call.
    Unanswered / blocked questions are handled without calling the LLM.

    Returns:
        (scores_map, blocked_map, final_report, warning)
        - scores_map:   question_id → AnswerScore
        - blocked_map:  question_id → reason string (safety block or not answered)
        - final_report: FinalReport assembled in Python (average computed here)
        - warning:      non-fatal warning string (empty if none)
    """
    warning = ""
    scores_map: dict[str, AnswerScore] = {}
    blocked_map: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # 1. Per-answer local guardrails — fill blocked_map for bad answers,
    #    items_to_send for answered + clean answers, score 0 for unanswered.
    # -----------------------------------------------------------------------
    items_to_send: list[BatchAnswerItem] = []

    for q in questions:
        qid = q.question_id
        raw_answer = answers.get(qid, "").strip()

        if not raw_answer:
            # Not answered — score 0 locally, no LLM call
            blocked_map[qid] = "Not answered"
            continue

        # Validate / sanitize length and control chars
        v = validate_answer(raw_answer)
        if not v.passed:
            blocked_map[qid] = f"Blocked by safety check: {v.message}"
            logger.warning("Answer blocked (validate_answer): qid=%s reason=%s", qid, v.message)
            continue

        clean_answer = v.modified_text or raw_answer

        # Injection check
        inj = check_injection(clean_answer, source="answer", session_id=session_id)
        if not inj.passed:
            blocked_map[qid] = "Blocked by safety check: injection detected"
            logger.warning("Answer blocked (injection): qid=%s", qid)
            continue

        # Content safety
        safety = check_content_safety(clean_answer, session_id=session_id)
        if not safety.passed:
            reason = "Blocked by safety check: self-harm signal" if safety.is_self_harm else "Blocked by safety check: unsafe content"
            blocked_map[qid] = reason
            logger.warning("Answer blocked (content_safety): qid=%s", qid)
            continue

        # Collect key points from retrieved sources
        key_points: list[str] = []
        for src_id in q.grounded_in_source_ids:
            for src in retrieved_sources:
                if src["id"] == src_id:
                    key_points.extend(src.get("key_points", []))
        if not key_points:
            key_points = ["Relevance to the question", "Technical depth", "Clear structure", "Specific examples"]

        items_to_send.append(
            BatchAnswerItem(
                question_id=qid,
                question_text=q.question_text,
                key_points=key_points,
                answer=clean_answer,
            )
        )

    # -----------------------------------------------------------------------
    # 2. If nothing to send (all unanswered / blocked) — skip LLM call
    # -----------------------------------------------------------------------
    if not items_to_send:
        # Build a zero-score report entirely locally
        for q in questions:
            qid = q.question_id
            if qid not in blocked_map:
                blocked_map[qid] = "Not answered"
        scores_map = _build_zero_scores(questions, blocked_map)
        report = _build_report_from_scores(questions, scores_map, [], [])
        return scores_map, blocked_map, report, warning

    # -----------------------------------------------------------------------
    # 3. Rate limit check for the batch scoring call
    # -----------------------------------------------------------------------
    rate_check = rate_limiter.check()
    if not rate_check.passed:
        raise LLMError(rate_check.message)

    # -----------------------------------------------------------------------
    # 4. Build the batch scoring prompt
    # -----------------------------------------------------------------------
    questions_and_answers = _format_batch_for_prompt(items_to_send)
    question_ids_str = ", ".join(item.question_id for item in items_to_send)

    system = SYSTEM_PROMPT_BATCH_SCORING
    user = USER_PROMPT_BATCH_SCORING.format(
        n_answered=len(items_to_send),
        role=role,
        questions_and_answers=questions_and_answers,
        question_ids=question_ids_str,
    )

    # -----------------------------------------------------------------------
    # 5. LLM call 2 — with retry and half-batch fallback on truncated JSON
    # -----------------------------------------------------------------------
    client: WatsonxLLMClient = get_llm_client()
    rate_limiter.record_call()

    batch_response = _call_batch_scoring_with_fallback(
        client=client,
        user=user,
        system=system,
        items=items_to_send,
        role=role,
    )

    # -----------------------------------------------------------------------
    # 6. Validate that we got exactly the expected question IDs
    # -----------------------------------------------------------------------
    submitted_ids = {item.question_id for item in items_to_send}
    returned_ids = {r.question_id for r in batch_response.results}
    missing_ids = submitted_ids - returned_ids

    if missing_ids:
        logger.warning("Batch scoring missing results for: %s", missing_ids)
        warning = (
            f"Scoring result was incomplete; {len(missing_ids)} question(s) "
            "received a score of 0."
        )

    # -----------------------------------------------------------------------
    # 7. Post-process results — output guardrails on each result
    # -----------------------------------------------------------------------
    for result in batch_response.results:
        qid = result.question_id
        if qid not in submitted_ids:
            # Extra / unexpected question_id — skip
            continue

        # Bias check and filter on feedback / model_answer
        result = _apply_output_guardrails(result)
        if result is None:
            blocked_map[qid] = "Blocked by output safety check"
            continue

        scores_map[qid] = result.to_answer_score()

    # Score 0 for questions whose results were missing from the response
    for qid in missing_ids:
        blocked_map[qid] = "No scoring result returned"

    # -----------------------------------------------------------------------
    # 8. Build the final report (average computed in Python)
    # -----------------------------------------------------------------------
    all_scores_map = _build_zero_scores(questions, blocked_map)
    all_scores_map.update(scores_map)

    report = _build_report_from_scores(
        questions=questions,
        scores_map=all_scores_map,
        strengths=batch_response.strengths[:3],
        improvements=batch_response.improvements[:3],
    )

    return scores_map, blocked_map, report, warning


# ---------------------------------------------------------------------------
# Batch scoring helpers
# ---------------------------------------------------------------------------

def _format_batch_for_prompt(items: list[BatchAnswerItem]) -> str:
    """Render all Q&A items for the batch prompt."""
    parts: list[str] = []
    for item in items:
        kp_text = "\n".join(f"  - {kp}" for kp in item.key_points) if item.key_points else "  - (general rubric)"
        parts.append(
            f"QUESTION [{item.question_id}]: {item.question_text}\n"
            f"KEY POINTS:\n{kp_text}\n"
            f"<answer id=\"{item.question_id}\">\n{item.answer}\n</answer>"
        )
    return "\n\n".join(parts)


def _call_batch_scoring_with_fallback(
    client: WatsonxLLMClient,
    user: str,
    system: str,
    items: list[BatchAnswerItem],
    role: str,
) -> BatchScoringResponse:
    """
    Try the full batch; on truncated JSON, retry once with repair prompt,
    then fall back to two smaller batches before giving up.
    """
    from src.llm import _extract_json  # module-level helper

    n = len(items)

    # First attempt: full batch
    try:
        raw_data = client.generate_batch_scoring(
            prompt=user,
            system_prompt=system,
            n_questions=n,
        )
        return BatchScoringResponse.model_validate(raw_data)
    except (LLMError, Exception) as exc:
        logger.warning("Batch scoring first attempt failed: %s", exc)

    # Retry with repair prompt (generate_json already does internal repair,
    # but here we catch pydantic failures after repair)
    try:
        raw_data = client.generate_batch_scoring(
            prompt=user,
            system_prompt=system,
            n_questions=n,
        )
        return BatchScoringResponse.model_validate(raw_data)
    except Exception as exc:
        logger.warning("Batch scoring retry also failed: %s", exc)

    # Fallback: split into two smaller batches
    if n <= 1:
        raise LLMError("Batch scoring failed for a single question. Please try again.")

    logger.info("Falling back to two smaller batches (%d items)", n)
    mid = n // 2
    first_half = items[:mid]
    second_half = items[mid:]

    results_all: list[Any] = []
    strengths: list[str] = []
    improvements: list[str] = []

    for chunk in (first_half, second_half):
        q_ids = ", ".join(c.question_id for c in chunk)
        chunk_prompt = USER_PROMPT_BATCH_SCORING.format(
            n_answered=len(chunk),
            role=role,
            questions_and_answers=_format_batch_for_prompt(chunk),
            question_ids=q_ids,
        )
        try:
            raw_data = client.generate_batch_scoring(
                prompt=chunk_prompt,
                system_prompt=system,
                n_questions=len(chunk),
            )
            partial = BatchScoringResponse.model_validate(raw_data)
            results_all.extend(partial.results)
            if not strengths:
                strengths = partial.strengths
            if not improvements:
                improvements = partial.improvements
        except Exception as exc:
            logger.error("Fallback chunk scoring also failed: %s", exc)
            raise LLMError(f"Batch scoring failed completely: {exc}") from exc

    # Merge the two halves into a single BatchScoringResponse
    merged_data = {
        "results": [r.model_dump() for r in results_all],
        "strengths": strengths,
        "improvements": improvements,
    }
    return BatchScoringResponse.model_validate(merged_data)


def _apply_output_guardrails(result: Any) -> Any | None:
    """
    Run output safety checks on feedback + model_answer.
    Returns updated result or None if it should be blocked entirely.
    """
    warning_issued = False
    for field_name in ("feedback", "model_answer"):
        text = getattr(result, field_name)

        # Bias check
        bias = check_bias_in_output(text)
        if not bias.passed:
            clean = filter_biased_sentences(text)
            result = result.model_copy(update={field_name: clean})

        # Secrets scan
        secrets = scan_for_secrets(text)
        if not secrets.passed:
            logger.error("Secrets found in scoring output, blocking result: qid=%s", result.question_id)
            return None

        # Output content / length check
        out_check = check_output(text, context="scoring_output")
        if not out_check.passed:
            logger.warning("Output check failed for qid=%s field=%s: %s", result.question_id, field_name, out_check.message)
            return None

    return result


def _build_zero_scores(
    questions: list[InterviewQuestion],
    blocked_map: dict[str, str],
) -> dict[str, AnswerScore]:
    """Create score=0 AnswerScore entries for all blocked / unanswered questions."""
    zero_scores: dict[str, AnswerScore] = {}
    for q in questions:
        qid = q.question_id
        if qid in blocked_map:
            reason = blocked_map[qid]
            zero_scores[qid] = AnswerScore(
                question_id=qid,
                score=0,
                feedback=reason,
                model_answer="No model answer available.",
                covered_points=[],
                missed_points=[],
            )
    return zero_scores


def _build_report_from_scores(
    questions: list[InterviewQuestion],
    scores_map: dict[str, AnswerScore],
    strengths: list[str],
    improvements: list[str],
) -> FinalReport:
    """Build the FinalReport; average score computed in Python."""
    per_q: list[dict[str, Any]] = []
    total_score = 0.0
    count = 0

    for q in questions:
        score_obj = scores_map.get(q.question_id)
        score_val = score_obj.score if score_obj else 0
        per_q.append(
            {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "score": score_val,
            }
        )
        total_score += score_val
        count += 1

    avg = round(total_score / count, 2) if count else 0.0

    return FinalReport(
        average_score=avg,
        per_question_scores=per_q,
        top_strengths=strengths if strengths else ["Attempted the interview"],
        top_improvements=improvements if improvements else ["Review the model answers"],
        summary="",
    )


# ---------------------------------------------------------------------------
# Legacy single-answer scoring (kept for backwards compatibility / tests)
# ---------------------------------------------------------------------------

def score_answer(
    question: InterviewQuestion,
    answer_text: str,
    key_points: list[str],
    rate_limiter: RateLimiter,
    session_id: str = "",
) -> tuple[AnswerScore, str]:
    """
    Score a single candidate answer using Granite (legacy single-call path).

    This function is kept for unit tests; the main app now uses score_all_answers.
    """
    warning = ""

    rate_check = rate_limiter.check()
    if not rate_check.passed:
        raise LLMError(rate_check.message)

    inj = check_injection(answer_text, source="answer", session_id=session_id)
    if not inj.passed:
        raise ValueError(inj.message)

    safety = check_content_safety(answer_text, session_id=session_id)
    if not safety.passed:
        raise ValueError(safety.message)

    system = SYSTEM_PROMPT_SCORING
    user = USER_PROMPT_SCORING.format(
        question_text=question.question_text,
        key_points="\n".join(f"- {kp}" for kp in key_points),
        answer_text=answer_text,
        question_id=question.question_id,
    )

    client: WatsonxLLMClient = get_llm_client()
    rate_limiter.record_call()

    raw_data = client.generate_scoring(prompt=user, system_prompt=system)

    try:
        score_result = AnswerScore.model_validate(raw_data)
    except Exception as exc:
        raise LLMError(f"Scoring returned an invalid schema: {exc}") from exc

    if score_result.question_id != question.question_id:
        score_result = score_result.model_copy(update={"question_id": question.question_id})

    for field_text in [score_result.feedback, score_result.model_answer]:
        bias = check_bias_in_output(field_text)
        if not bias.passed:
            clean = filter_biased_sentences(field_text)
            if field_text == score_result.feedback:
                score_result = score_result.model_copy(update={"feedback": clean})
            else:
                score_result = score_result.model_copy(update={"model_answer": clean})
            if not warning:
                warning = "Some feedback was filtered for fairness."

        secrets = scan_for_secrets(field_text)
        if not secrets.passed:
            raise LLMError(secrets.message)

        out_check = check_output(field_text, context="scoring_output")
        if not out_check.passed:
            raise LLMError(out_check.message)

    guardian_result = client.guardian_check(
        score_result.feedback + " " + score_result.model_answer
    )
    if not guardian_result.get("safe", True):
        raise LLMError(
            "The AI feedback did not pass safety verification. Please try again."
        )

    return score_result, warning


# ---------------------------------------------------------------------------
# Final report generation (legacy path; kept for backwards compat)
# ---------------------------------------------------------------------------

def generate_report(
    role: str,
    questions: list[InterviewQuestion],
    scores: list[AnswerScore],
    rate_limiter: RateLimiter,
    session_id: str = "",
) -> FinalReport:
    """Generate the final performance report (legacy single call)."""
    rate_check = rate_limiter.check()
    if not rate_check.passed:
        raise LLMError(rate_check.message)

    scores_data = []
    for q, s in zip(questions, scores):
        scores_data.append(
            {
                "question_id": q.question_id,
                "question_text": q.question_text[:200],
                "score": s.score,
                "feedback_summary": s.feedback[:300],
                "covered_points": s.covered_points[:3],
                "missed_points": s.missed_points[:3],
            }
        )

    scores_json_str = json.dumps(scores_data, indent=2)

    system = SYSTEM_PROMPT_REPORT
    user = USER_PROMPT_REPORT.format(role=role, scores_json=scores_json_str)

    client: WatsonxLLMClient = get_llm_client()
    rate_limiter.record_call()

    raw_data = client.generate_report(prompt=user, system_prompt=system)

    avg = sum(s.score for s in scores) / len(scores) if scores else 0.0

    try:
        report = FinalReport.model_validate(raw_data)
    except Exception as exc:
        logger.warning("Report schema validation failed (%s), building fallback report", exc)
        report = FinalReport(
            average_score=round(avg, 2),
            per_question_scores=[
                {
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "score": s.score,
                }
                for q, s in zip(questions, scores)
            ],
            top_strengths=["Attempted all questions", "Showed knowledge in some areas"],
            top_improvements=["Provide more specific examples", "Cover more key points"],
            summary="Performance report generated with limited AI analysis.",
        )

    report = report.model_copy(update={"average_score": round(avg, 2)})

    bias = check_bias_in_output(report.summary)
    if not bias.passed:
        report = report.model_copy(update={"summary": filter_biased_sentences(report.summary)})

    return report
