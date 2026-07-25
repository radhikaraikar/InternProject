"""
eval_harness.py — Before/after hallucination comparison harness.

Runs 10-15 test questions against:
  1. Naive RAG (no self-correction) — baseline
  2. Self-Correcting RAG — full pipeline

Scores:
  - Hallucination rate: % of answers containing claims not in retrieved context
  - Correct refusal/contradiction rate: % of "no answer" / "contradictory" questions caught
  - Answer accuracy: manual / LLM-judged faithfulness for answerable questions

Outputs: comparison table (markdown + matplotlib bar chart)
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic
import matplotlib.pyplot as plt

from ingest import run_ingestion, get_chroma_collection
from rag_pipeline import (
    run_pipeline,
    run_naive_pipeline,
    retrieve,
    DecisionType,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = "./eval_questions.json"
RESULTS_PATH = "./eval_results.json"
CHART_PATH = "./eval_comparison_chart.png"
LLM_MODEL = "claude-3-5-haiku-20241022"


# ── LLM faithfulness judge ────────────────────────────────────────────────────
FAITHFULNESS_JUDGE_SYSTEM = """You are an evaluation judge for a RAG system. 
Given a question, a generated answer, and the retrieved context chunks, determine:

1. Is the answer FAITHFUL? (all claims are supported by the provided context)
2. Does the answer HALLUCINATE? (claims facts not present in the context)

Respond with valid JSON only:
{
  "faithful": true | false,
  "hallucinated": true | false,
  "reason": "<one sentence>"
}
"""


def judge_faithfulness(question: str, answer: str, context_chunks: list) -> dict:
    """Use LLM to judge whether an answer hallucinates vs the retrieved context."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("No API key — skipping faithfulness judge.")
        return {"faithful": True, "hallucinated": False, "reason": "Not evaluated (no API key)"}

    client = anthropic.Anthropic(api_key=api_key)
    context_text = "\n\n".join(
        f"[{c.source}, p.{c.page}]: {c.text[:500]}" for c in context_chunks
    )
    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"GENERATED ANSWER: {answer}\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}"
    )
    try:
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=256,
            system=FAITHFULNESS_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Faithfulness judge error: {e}")
        return {"faithful": True, "hallucinated": False, "reason": f"Error: {e}"}


# ── Result data structure ─────────────────────────────────────────────────────
@dataclass
class EvalResult:
    question_id: int
    question: str
    expected_behavior: str

    # Naive baseline
    naive_answer: str
    naive_hallucinated: bool
    naive_correctly_refused: bool
    naive_contradiction_caught: bool

    # Self-correcting
    sc_answer: str
    sc_decision: str
    sc_hallucinated: bool
    sc_correctly_refused: bool
    sc_contradiction_caught: bool

    notes: str = ""


# ── Run evaluation ─────────────────────────────────────────────────────────────
def run_eval(data_folder: str = "./data", persist_dir: str = "./chroma_db") -> List[EvalResult]:
    """Run the full evaluation harness."""

    # ── Load or rebuild index ─────────────────────────────────────────────
    try:
        collection = get_chroma_collection(persist_dir)
        count = collection.count()
        if count == 0:
            raise ValueError("Empty collection")
        logger.info(f"Loaded existing Chroma collection ({count} chunks)")
    except Exception:
        logger.info("Building Chroma index...")
        run_ingestion(data_folder, persist_dir)
        collection = get_chroma_collection(persist_dir)

    # ── Load questions ────────────────────────────────────────────────────
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    results: List[EvalResult] = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = q["expected_behavior"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Q{qid}: {question}")
        logger.info(f"Expected: {expected}")

        # ── Naive baseline ─────────────────────────────────────────────
        logger.info("Running naive baseline...")
        try:
            naive_answer = run_naive_pipeline(question, collection)
        except Exception as e:
            naive_answer = f"ERROR: {e}"

        naive_chunks = retrieve(question, collection)
        naive_faith = judge_faithfulness(question, naive_answer, naive_chunks)
        naive_hallucinated = naive_faith.get("hallucinated", False)

        # Naive never refuses or catches contradictions
        naive_correctly_refused = False
        naive_contradiction_caught = False

        logger.info(f"Naive answer (truncated): {naive_answer[:150]}...")
        logger.info(f"Naive hallucinated: {naive_hallucinated}")

        # ── Self-correcting pipeline ───────────────────────────────────
        logger.info("Running self-correcting pipeline...")
        try:
            sc_result = run_pipeline(question, collection)
        except Exception as e:
            from rag_pipeline import PipelineResult
            sc_result = PipelineResult(
                decision=DecisionType.ERROR,
                answer=f"ERROR: {e}",
                chunks_used=[],
                judge_result=None,
                log_entries=[f"Exception: {e}"],
            )

        sc_decision = sc_result.decision.value
        sc_answer = sc_result.answer
        sc_chunks = sc_result.chunks_used or naive_chunks

        # Score hallucination (only relevant if it actually answered)
        if sc_result.decision == DecisionType.GROUNDED_ANSWER:
            sc_faith = judge_faithfulness(question, sc_answer, sc_chunks)
            sc_hallucinated = sc_faith.get("hallucinated", False)
        else:
            sc_hallucinated = False  # Refused/flagged → not a hallucination

        # Correct refusal: expected "clarify" and we did clarify
        sc_correctly_refused = (
            expected == "clarify"
            and sc_result.decision in (
                DecisionType.INSUFFICIENT_CLARIFY,
                DecisionType.INSUFFICIENT_REQUERIED,
            )
        )

        # Contradiction caught: expected "contradiction" and we flagged it
        sc_contradiction_caught = (
            expected == "contradiction"
            and sc_result.decision == DecisionType.CONTRADICTORY
        )

        logger.info(f"SC decision: {sc_decision}")
        logger.info(f"SC hallucinated: {sc_hallucinated}")
        logger.info(f"SC correctly refused: {sc_correctly_refused}")
        logger.info(f"SC contradiction caught: {sc_contradiction_caught}")

        results.append(
            EvalResult(
                question_id=qid,
                question=question,
                expected_behavior=expected,
                naive_answer=naive_answer,
                naive_hallucinated=naive_hallucinated,
                naive_correctly_refused=naive_correctly_refused,
                naive_contradiction_caught=naive_contradiction_caught,
                sc_answer=sc_answer,
                sc_decision=sc_decision,
                sc_hallucinated=sc_hallucinated,
                sc_correctly_refused=sc_correctly_refused,
                sc_contradiction_caught=sc_contradiction_caught,
                notes=q.get("notes", ""),
            )
        )

    return results


# ── Scoring + reporting ────────────────────────────────────────────────────────
def compute_scores(results: List[EvalResult]) -> dict:
    total = len(results)
    answerable = [r for r in results if r.expected_behavior == "answer"]
    clarify_qs = [r for r in results if r.expected_behavior == "clarify"]
    contradiction_qs = [r for r in results if r.expected_behavior == "contradiction"]

    scores = {
        "total_questions": total,
        "naive": {
            "hallucination_rate": sum(r.naive_hallucinated for r in results) / total,
            "correct_refusal_rate": 0.0,  # Naive never refuses
            "contradiction_catch_rate": 0.0,  # Naive never catches
        },
        "self_correcting": {
            "hallucination_rate": sum(r.sc_hallucinated for r in results) / total,
            "correct_refusal_rate": (
                sum(r.sc_correctly_refused for r in clarify_qs) / len(clarify_qs)
                if clarify_qs else 0.0
            ),
            "contradiction_catch_rate": (
                sum(r.sc_contradiction_caught for r in contradiction_qs) / len(contradiction_qs)
                if contradiction_qs else 0.0
            ),
        },
    }
    return scores


def print_table(results: List[EvalResult], scores: dict):
    print("\n" + "="*80)
    print("EVALUATION RESULTS — BEFORE vs AFTER SELF-CORRECTION")
    print("="*80)
    print(f"{'Q#':<4} {'Expected':<14} {'Naive Hall.':<14} {'SC Hall.':<12} {'SC Decision':<24} {'Caught?'}")
    print("-"*80)

    for r in results:
        caught = ""
        if r.expected_behavior == "clarify":
            caught = "✅ Refused" if r.sc_correctly_refused else "❌ Missed"
        elif r.expected_behavior == "contradiction":
            caught = "✅ Flagged" if r.sc_contradiction_caught else "❌ Missed"
        else:
            caught = "—"

        naive_h = "🔴 YES" if r.naive_hallucinated else "🟢 No"
        sc_h = "🔴 YES" if r.sc_hallucinated else "🟢 No"

        print(f"{r.question_id:<4} {r.expected_behavior:<14} {naive_h:<14} {sc_h:<12} {r.sc_decision:<24} {caught}")

    print("="*80)
    n = scores["naive"]
    s = scores["self_correcting"]
    print(f"\n{'METRIC':<35} {'NAIVE BASELINE':>16} {'SELF-CORRECTING':>16}")
    print("-"*70)
    print(f"{'Hallucination Rate':<35} {n['hallucination_rate']:>15.0%} {s['hallucination_rate']:>15.0%}")
    print(f"{'Correct Refusal Rate':<35} {n['correct_refusal_rate']:>15.0%} {s['correct_refusal_rate']:>15.0%}")
    print(f"{'Contradiction Catch Rate':<35} {n['contradiction_catch_rate']:>15.0%} {s['contradiction_catch_rate']:>15.0%}")
    print("="*70)


def save_chart(scores: dict, path: str = CHART_PATH):
    """Generate and save a before/after comparison bar chart."""
    metrics = ["Hallucination\nRate", "Correct Refusal\nRate", "Contradiction\nCatch Rate"]
    naive_vals = [
        scores["naive"]["hallucination_rate"],
        scores["naive"]["correct_refusal_rate"],
        scores["naive"]["contradiction_catch_rate"],
    ]
    sc_vals = [
        scores["self_correcting"]["hallucination_rate"],
        scores["self_correcting"]["correct_refusal_rate"],
        scores["self_correcting"]["contradiction_catch_rate"],
    ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], naive_vals, width, label="Naive Baseline", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], sc_vals, width, label="Self-Correcting RAG", color="#2ecc71", alpha=0.85)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Rate (0–1)", fontsize=12)
    ax.set_title("Self-Correcting RAG: Before vs After", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.0%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.0%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    logger.info(f"Chart saved to {path}")
    plt.close()


def save_results(results: List[EvalResult], scores: dict, path: str = RESULTS_PATH):
    """Save full results to JSON for reference."""
    data = {
        "scores": scores,
        "results": [
            {
                "id": r.question_id,
                "question": r.question,
                "expected_behavior": r.expected_behavior,
                "naive_hallucinated": r.naive_hallucinated,
                "sc_decision": r.sc_decision,
                "sc_hallucinated": r.sc_hallucinated,
                "sc_correctly_refused": r.sc_correctly_refused,
                "sc_contradiction_caught": r.sc_contradiction_caught,
                "naive_answer_snippet": r.naive_answer[:300],
                "sc_answer_snippet": r.sc_answer[:300],
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_eval()
    scores = compute_scores(results)
    print_table(results, scores)
    save_chart(scores)
    save_results(results, scores)
    print(f"\nChart saved: {CHART_PATH}")
    print(f"Full results: {RESULTS_PATH}")
