"""
Interview Trainer — Streamlit UI (batch flow).

Strictly UI only: reads/writes session state and calls src modules.
No business logic lives here.

Batch flow (2 LLM calls per interview session):
  Step 1 — Resume upload → guardrails → "Generate Interview" button.
  Call 1 — generate_questions: ONE LLM call returns all N questions at once.
  Step 2 — Answer cards rendered in a single st.form; progress counter shown.
  Call 2 — score_all_answers: ONE LLM call scores all submitted answers and
            returns the full report structure.
  Step 3 — Final report: average score, bar chart, strengths, improvements.
"""
from __future__ import annotations

import os
import sys
from typing import Any

# Ensure project root is on the path when running as `streamlit run app.py`
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Interview Trainer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Deferred imports (config validation happens here)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to watsonx.ai…")
def _get_llm():
    from src.llm import get_llm_client
    return get_llm_client()


@st.cache_resource(show_spinner="Building question index…")
def _build_rag_index():
    from src.rag import build_index
    build_index()
    return True


def _init_services() -> tuple[bool, str]:
    """Try to initialise LLM and RAG; return (ok, error_message)."""
    try:
        from src.config import (  # noqa: F401 – validates env vars at import
            WATSONX_API_KEY, WATSONX_PROJECT_ID
        )
    except SystemExit as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Configuration error: {exc}"

    try:
        _build_rag_index()
    except Exception as exc:
        return False, f"RAG index build failed: {exc}"

    llm = _get_llm()
    if not llm.is_available:
        return False, (
            "Could not connect to watsonx.ai. "
            "Check your API key and project ID in .env."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults: dict = {
        "step": "resume",           # resume | generating | answers | scoring | report
        "resume_text": "",
        "questions": [],            # list[InterviewQuestion]
        "retrieved_sources": [],
        "answers": {},              # question_id → answer text (saved on submit)
        "scores": {},               # question_id → AnswerScore
        "blocked": {},              # question_id → reason string
        "final_report": None,
        "session_id": _new_session_id(),
        "rate_limiter": None,
        "bad_reports": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state["rate_limiter"] is None:
        from src.guardrails import RateLimiter
        st.session_state["rate_limiter"] = RateLimiter()


def _new_session_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _clear_session() -> None:
    """Wipe all session data (privacy clear button)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _reset_for_new_interview() -> None:
    """Clear interview data but preserve the sidebar settings."""
    for key in [
        "step", "resume_text", "questions", "retrieved_sources",
        "answers", "scores", "blocked", "final_report",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    from src.guardrails import RateLimiter
    st.session_state["rate_limiter"] = RateLimiter()
    st.session_state["session_id"] = _new_session_id()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> tuple[str, str, int, bool]:
    with st.sidebar:
        st.title("⚙️ Settings")

        from src.config import AVAILABLE_ROLES, AVAILABLE_INTERVIEW_TYPES, MIN_QUESTIONS, MAX_QUESTIONS

        role = st.selectbox("Target role", AVAILABLE_ROLES, key="sidebar_role")
        interview_type = st.selectbox(
            "Interview type",
            AVAILABLE_INTERVIEW_TYPES,
            key="sidebar_itype",
        )
        n_questions = st.slider(
            "Number of questions",
            min_value=MIN_QUESTIONS,
            max_value=MAX_QUESTIONS,
            value=5,
            key="sidebar_nq",
        )
        redact_toggle = st.toggle(
            "Redact contact details before sending to AI",
            value=True,
            key="sidebar_redact",
        )

        st.divider()

        st.markdown(
            "**🔒 Privacy notice**\n\n"
            "Your resume and answers are sent to IBM watsonx.ai for analysis. "
            "Nothing is stored to disk. Data lives only in your browser session. "
            "Use the button below to erase everything."
        )
        if st.button("🗑️ Clear my data", use_container_width=True):
            _clear_session()

        st.divider()
        st.caption("⚠️ AI-generated feedback is for **practice only** and is not a hiring decision.")

    return role, interview_type, n_questions, redact_toggle


# ---------------------------------------------------------------------------
# Step 1: Resume upload
# ---------------------------------------------------------------------------

def _render_resume_step(role: str, interview_type: str, n_questions: int, redact_pii: bool) -> None:
    st.header("📄 Step 1: Upload Your Resume")
    st.markdown("Paste your resume text below, or upload a PDF / TXT file (max 2 MB).")

    col1, col2 = st.columns([2, 1])

    with col1:
        resume_text = st.text_area(
            "Resume text",
            value=st.session_state.get("resume_text", ""),
            height=300,
            placeholder="Paste your resume here…",
            key="resume_input",
        )

    with col2:
        uploaded_file = st.file_uploader(
            "Or upload a file",
            type=["pdf", "txt"],
            key="file_uploader",
        )
        if uploaded_file is not None:
            from src.guardrails import validate_file_upload

            result = validate_file_upload(
                filename=uploaded_file.name,
                content=uploaded_file.read(),
                mime_type=uploaded_file.type or "",
            )
            if result.passed:
                resume_text = result.modified_text or ""
                st.success("File uploaded and text extracted.")
            else:
                st.error(result.message)
                return

    if st.button("Generate Interview →", type="primary", use_container_width=True):
        from src.guardrails import (
            validate_resume,
            check_injection,
            check_content_safety,
            check_language,
        )

        v = validate_resume(resume_text)
        if not v.passed:
            st.error(v.message)
            return

        clean_text = v.modified_text or resume_text

        inj = check_injection(clean_text, session_id=st.session_state["session_id"])
        if not inj.passed:
            st.error(inj.message)
            return

        safety = check_content_safety(clean_text, session_id=st.session_state["session_id"])
        if not safety.passed:
            if safety.is_self_harm:
                st.warning(safety.message)
            else:
                st.error(safety.message)
            return

        lang = check_language(clean_text)
        if lang.message:
            st.warning(lang.message)

        st.session_state["resume_text"] = clean_text
        st.session_state["step"] = "generating"
        st.rerun()


# ---------------------------------------------------------------------------
# Question generation — LLM call 1
# ---------------------------------------------------------------------------

def _generate_questions_step(role: str, interview_type: str, n_questions: int, redact_pii: bool) -> None:
    st.header("🔄 Generating your questions…")

    with st.spinner("Calling watsonx.ai to generate all interview questions (call 1 of 2)…"):
        from src import agent as agent_module

        try:
            questions, retrieved, warning = agent_module.generate_questions(
                resume_text=st.session_state["resume_text"],
                role=role,
                interview_type=interview_type,
                n_questions=n_questions,
                redact_pii=redact_pii,
                rate_limiter=st.session_state["rate_limiter"],
                session_id=st.session_state["session_id"],
            )
            st.session_state["questions"] = questions
            st.session_state["retrieved_sources"] = retrieved
            st.session_state["answers"] = {}
            st.session_state["scores"] = {}
            st.session_state["blocked"] = {}

            if warning:
                st.warning(warning)

            st.session_state["step"] = "answers"
            st.rerun()

        except Exception as exc:
            st.error(f"Could not generate questions: {exc}")
            if st.button("← Back to resume"):
                st.session_state["step"] = "resume"
                st.rerun()


# ---------------------------------------------------------------------------
# Step 2: Answer all questions at once (single form, no per-question rerun)
# ---------------------------------------------------------------------------

def _render_answers_step() -> None:
    questions = st.session_state["questions"]
    total = len(questions)

    st.header("💬 Step 2: Answer All Questions")
    st.caption("🤖 AI-generated questions")
    _render_safety_badge(passed=True)

    st.info(f"**{total} question(s)** — fill in as many as you can, then click Submit.")

    # All answer cards inside one form so typing does not trigger reruns
    with st.form(key="answers_form"):
        form_answers: dict[str, str] = {}

        for i, q in enumerate(questions):
            diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")

            with st.container():
                st.markdown(f"---")
                col_badge, col_title = st.columns([1, 6])
                with col_badge:
                    st.markdown(f"**Q{i+1}**")
                    st.caption(f"{diff_color} {q.difficulty.capitalize()}")
                with col_title:
                    st.markdown(f"**{q.question_text}**")
                    st.caption("🤖 AI-generated question")

                if q.is_general_fallback:
                    st.info("ℹ️ This is a general role-based question (no close match in the question bank).")

                # "Why this question?" expander
                with st.expander("🔍 Why this question?"):
                    if q.grounded_in_source_ids:
                        st.write(f"**Grounded in source question IDs:** {', '.join(q.grounded_in_source_ids)}")
                    if q.resume_context_hint:
                        st.write(f"**Resume context:** {q.resume_context_hint}")
                    sources = st.session_state.get("retrieved_sources", [])
                    relevant = [s for s in sources if s["id"] in q.grounded_in_source_ids]
                    if relevant:
                        for s in relevant:
                            st.markdown(f"- **[{s['id']}]** {s['question']}")
                    else:
                        st.write("No direct source reference available.")

                # Answer text area — inside the form (no rerun on keystrokes)
                prev = st.session_state.get("answers", {}).get(q.question_id, "")
                answer = st.text_area(
                    f"Your answer to Q{i+1}",
                    value=prev,
                    height=150,
                    placeholder="Type your answer here… (10–2000 characters)",
                    key=f"form_answer_{q.question_id}",
                )
                form_answers[q.question_id] = answer

        st.markdown("---")

        submitted = st.form_submit_button(
            "Submit all answers →",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        st.session_state["answers"] = form_answers
        st.session_state["step"] = "scoring"
        st.rerun()


# ---------------------------------------------------------------------------
# Batch scoring — LLM call 2
# ---------------------------------------------------------------------------

def _score_answers_step(role: str) -> None:
    st.header("📊 Scoring your answers…")

    with st.spinner("Calling watsonx.ai to score all answers together (call 2 of 2)…"):
        from src import agent as agent_module

        questions = st.session_state["questions"]
        answers = st.session_state["answers"]
        retrieved = st.session_state["retrieved_sources"]

        try:
            scores_map, blocked_map, report, warning = agent_module.score_all_answers(
                questions=questions,
                answers=answers,
                retrieved_sources=retrieved,
                role=role,
                rate_limiter=st.session_state["rate_limiter"],
                session_id=st.session_state["session_id"],
            )
            st.session_state["scores"] = scores_map
            st.session_state["blocked"] = blocked_map
            st.session_state["final_report"] = report

            if warning:
                st.warning(warning)

            st.session_state["step"] = "report"
            st.rerun()

        except Exception as exc:
            st.error(f"Could not score answers: {exc}")
            if st.button("← Back to answers"):
                st.session_state["step"] = "answers"
                st.rerun()


# ---------------------------------------------------------------------------
# Step 3: Final report
# ---------------------------------------------------------------------------

def _render_report_step(role: str) -> None:
    report = st.session_state.get("final_report")
    if report is None:
        st.error("No report found. Please complete the interview first.")
        return

    questions = st.session_state["questions"]
    scores_map: dict[str, Any] = st.session_state["scores"]
    blocked_map: dict[str, str] = st.session_state.get("blocked", {})

    st.header("📊 Step 3: Your Performance Report")
    st.caption("🤖 AI-generated report")
    _render_safety_badge(passed=True)

    st.info(
        "⚠️ **Disclaimer:** AI-generated feedback is for practice purposes only, "
        "may be imperfect, and is **not** a hiring decision."
    )

    # -----------------------------------------------------------------------
    # Top metrics
    # -----------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{report.average_score:.1f} / 10")
    with col2:
        answered_count = sum(
            1 for q in questions
            if q.question_id in scores_map
        )
        st.metric("Questions Scored", f"{answered_count} / {len(questions)}")
    with col3:
        rl: Any = st.session_state["rate_limiter"]
        st.metric("AI Calls Remaining", rl.calls_remaining)

    st.divider()

    # -----------------------------------------------------------------------
    # Per-question bar chart
    # -----------------------------------------------------------------------
    if report.per_question_scores:
        import pandas as pd

        chart_data = {
            "Question": [
                (
                    f"Q{i+1}: {item['question_text'][:40]}…"
                    if len(str(item.get("question_text", ""))) > 40
                    else f"Q{i+1}: {item.get('question_text', item.get('question_id', ''))}"
                )
                for i, item in enumerate(report.per_question_scores)
            ],
            "Score": [int(item["score"]) for item in report.per_question_scores],
        }
        df = pd.DataFrame(chart_data).set_index("Question")
        st.subheader("📈 Score by Question")
        st.bar_chart(df, y="Score", use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Strengths and improvements
    # -----------------------------------------------------------------------
    col_s, col_i = st.columns(2)
    with col_s:
        st.subheader("💪 Top Strengths")
        for strength in report.top_strengths:
            st.markdown(f"✅ {strength}")
    with col_i:
        st.subheader("🎯 Areas to Improve")
        for improvement in report.top_improvements:
            st.markdown(f"📌 {improvement}")

    st.divider()

    # -----------------------------------------------------------------------
    # Per-question detailed breakdown
    # -----------------------------------------------------------------------
    with st.expander("📖 Detailed per-question breakdown", expanded=False):
        for i, q in enumerate(questions):
            qid = q.question_id
            score = scores_map.get(qid)
            blocked_reason = blocked_map.get(qid, "")

            st.markdown(f"**Q{i+1}: {q.question_text}**")

            if score is not None:
                diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
                st.caption(f"Difficulty: {diff_color} {q.difficulty.capitalize()}")
                st.metric(label=f"Q{i+1} Score", value=f"{score.score}/10", label_visibility="collapsed")
                st.progress(score.score * 10)

                with st.expander(f"📝 Feedback (Q{i+1})"):
                    st.write(score.feedback)

                with st.expander(f"✅ Model answer (Q{i+1})"):
                    st.write(score.model_answer)

                if score.covered_points:
                    with st.expander(f"✔️ Covered points (Q{i+1})"):
                        for pt in score.covered_points:
                            st.markdown(f"- {pt}")

                if score.missed_points:
                    with st.expander(f"❌ Missed points (Q{i+1})"):
                        for pt in score.missed_points:
                            st.markdown(f"- {pt}")

            elif blocked_reason:
                st.warning(f"🚫 {blocked_reason}")
            else:
                st.info("No score recorded for this question.")

            st.markdown("---")

    # -----------------------------------------------------------------------
    # Report a bad response
    # -----------------------------------------------------------------------
    st.subheader("🚩 Report a bad response")
    bad_q_options = {
        f"Q{i+1}: {q.question_text[:60]}…": q.question_id
        for i, q in enumerate(questions)
    }
    selected_q_label = st.selectbox("Which question?", list(bad_q_options.keys()), key="bad_q_sel")
    bad_reason = st.text_input("Briefly describe the issue:", key="bad_reason")
    if st.button("Submit Report"):
        from src.logging_utils import log_bad_response_report
        qid = bad_q_options.get(selected_q_label, "unknown")
        log_bad_response_report(
            session_id=st.session_state["session_id"],
            question_id=qid,
            reason=bad_reason,
        )
        st.success("Report submitted. Thank you for the feedback.")

    st.divider()

    if st.button("🔄 Start new interview", type="primary", use_container_width=True):
        _reset_for_new_interview()


# ---------------------------------------------------------------------------
# Safety badge helper
# ---------------------------------------------------------------------------

def _render_safety_badge(passed: bool = True) -> None:
    if passed:
        st.caption("🛡️ Safety checks: **passed**")
    else:
        st.caption("⚠️ Safety checks: **review needed**")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    role, interview_type, n_questions, redact_pii = _render_sidebar()

    service_ok, service_err = _init_services()
    if not service_ok:
        st.error(
            f"**Interview Trainer could not start.**\n\n{service_err}\n\n"
            "Please check your `.env` file and restart."
        )
        st.stop()

    step = st.session_state.get("step", "resume")

    if step == "resume":
        _render_resume_step(role, interview_type, n_questions, redact_pii)

    elif step == "generating":
        _generate_questions_step(role, interview_type, n_questions, redact_pii)

    elif step == "answers":
        _render_answers_step()

    elif step == "scoring":
        _score_answers_step(role)

    elif step == "report":
        _render_report_step(role)

    else:
        st.error(f"Unknown step: {step}")
        st.session_state["step"] = "resume"
        st.rerun()


if __name__ == "__main__":
    main()
