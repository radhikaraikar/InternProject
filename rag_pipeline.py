"""
rag_pipeline.py — Retrieval + Self-Correction Logic

Pipeline stages:
  A. Retrieve top-k chunks + similarity scores
  B. Confidence gate (score threshold check)
  C. Contradiction detection via LLM judge
  D. Branch: answer / re-query / clarify / surface conflict
  E. Final grounded answer generation with citations
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import anthropic
import chromadb

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
TOP_K = 8                 # retrieve more so all 3 docs get a chance to surface
CONFIDENCE_THRESHOLD = 0.40   # lower threshold — distilbert scores compress toward center
MAX_REQUERY_ATTEMPTS = 1
LLM_MODEL = "claude-3-5-haiku-20241022"


# ── Data structures ──────────────────────────────────────────────────────────
class DecisionType(str, Enum):
    GROUNDED_ANSWER = "grounded_answer"
    INSUFFICIENT_REQUERIED = "insufficient_requeried"
    INSUFFICIENT_CLARIFY = "insufficient_clarify"
    CONTRADICTORY = "contradictory"
    ERROR = "error"


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float   # cosine similarity [0,1], higher = more relevant
    chunk_id: str


@dataclass
class JudgeResult:
    sufficient: bool
    contradictory: bool
    reasoning: str
    conflicting_claims: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    decision: DecisionType
    answer: str
    chunks_used: List[RetrievedChunk]
    judge_result: Optional[JudgeResult]
    requery_attempted: bool = False
    reformulated_query: Optional[str] = None
    confidence_score: float = 0.0
    log_entries: List[str] = field(default_factory=list)


# ── LLM client ───────────────────────────────────────────────────────────────
def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")
    return anthropic.Anthropic(api_key=api_key)


def _call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    client = _get_client()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


# ── Step A — Retrieval ────────────────────────────────────────────────────────
def retrieve(query: str, collection: chromadb.Collection, k: int = TOP_K) -> List[RetrievedChunk]:
    """Query ChromaDB and return chunks with cosine similarity scores."""
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[RetrievedChunk] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma cosine distance ∈ [0,2]; convert to similarity [0,1]
        similarity = max(0.0, 1.0 - dist / 2.0)
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=round(similarity, 4),
                chunk_id=meta.get("chunk_id", ""),
            )
        )

    return chunks


def retrieve_cross_source(
    query: str, collection: chromadb.Collection, k: int = TOP_K
) -> List[RetrievedChunk]:
    """
    Retrieve top-k chunks then ensure the best chunk from every distinct
    source is included — so the judge always sees cross-document context.
    """
    # Fetch more than k to have material to sample from
    fetch_n = min(collection.count(), max(k, 15))
    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    all_chunks: List[RetrievedChunk] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = max(0.0, 1.0 - dist / 2.0)
        all_chunks.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=round(similarity, 4),
                chunk_id=meta.get("chunk_id", ""),
            )
        )

    # Best chunk per source (preserving score order within each source)
    seen_sources: dict = {}
    for c in all_chunks:
        if c.source not in seen_sources:
            seen_sources[c.source] = c

    # Final list: top-k by score, but always include the best per-source chunk
    top_k = all_chunks[:k]
    top_k_ids = {c.chunk_id for c in top_k}
    for best in seen_sources.values():
        if best.chunk_id not in top_k_ids:
            top_k.append(best)

    # Re-sort by score descending
    top_k.sort(key=lambda c: c.score, reverse=True)
    return top_k


# ── Step B — Confidence gate ──────────────────────────────────────────────────
def check_confidence(chunks: List[RetrievedChunk]) -> Tuple[bool, float]:
    """
    Returns (is_sufficient, top_score).
    Sufficient if the best chunk clears the threshold.
    """
    if not chunks:
        return False, 0.0
    top_score = chunks[0].score
    return top_score >= CONFIDENCE_THRESHOLD, top_score


# ── Step C — LLM contradiction/sufficiency judge ─────────────────────────────
JUDGE_SYSTEM = """You are a strict fact-checking judge. Given a question and a set of retrieved context chunks, 
you must determine:
1. Whether the chunks together contain SUFFICIENT information to answer the question.
2. Whether any chunks CONTRADICT each other on factual claims.

You MUST respond with valid JSON only — no prose, no markdown fences. Schema:
{
  "sufficient": true | false,
  "contradictory": true | false,
  "reasoning": "<one sentence>",
  "conflicting_claims": ["<claim from source A>", "<claim from source B>"]
}

Rules:
- sufficient=true only if the chunks explicitly state the information needed to answer.
- contradictory=true only if two or more chunks make DIRECTLY opposing factual claims.
- conflicting_claims should list the conflicting statements verbatim (up to 2).
- If not contradictory, conflicting_claims = []
"""


def judge_context(query: str, chunks: List[RetrievedChunk]) -> JudgeResult:
    """Ask the LLM judge to assess sufficiency and contradictions."""
    context_block = "\n\n".join(
        f"[Chunk {i+1} | {c.source} p.{c.page} | score={c.score:.3f}]\n{c.text}"
        for i, c in enumerate(chunks)
    )
    user_prompt = f"QUESTION: {query}\n\nCONTEXT CHUNKS:\n{context_block}"

    try:
        raw = _call_llm(JUDGE_SYSTEM, user_prompt, max_tokens=512)
        # Strip markdown fences if the model wraps in ```json
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
        return JudgeResult(
            sufficient=bool(data.get("sufficient", False)),
            contradictory=bool(data.get("contradictory", False)),
            reasoning=data.get("reasoning", ""),
            conflicting_claims=data.get("conflicting_claims", []),
        )
    except Exception as e:
        logger.warning(f"Judge parsing failed: {e}. Defaulting to sufficient=True.")
        return JudgeResult(sufficient=True, contradictory=False, reasoning=f"Judge error: {e}")


# ── Step D helper — Query reformulation ──────────────────────────────────────
REFORMULATE_SYSTEM = """You are a search query optimizer. Given a question that did not retrieve 
useful results, generate ONE alternative rephrase that might retrieve better information.
Output only the rephrased question — no explanation, no numbering."""


def reformulate_query(original_query: str) -> str:
    """Generate an alternative query phrasing."""
    try:
        return _call_llm(REFORMULATE_SYSTEM, f"Original question: {original_query}", max_tokens=128)
    except Exception as e:
        logger.warning(f"Query reformulation failed: {e}")
        return original_query


# ── Step E — Final answer generation ─────────────────────────────────────────
ANSWER_SYSTEM = """You are a precise, citation-driven research assistant.

RULES (follow strictly):
1. Answer ONLY using the provided context chunks. Do NOT use outside knowledge.
2. If the context does not fully support an answer, explicitly say so.
3. Cite every factual claim with [source_filename, page_N].
4. Be concise. Do not pad the answer with filler phrases.
5. If the context partially answers but leaves gaps, state what is and is not covered.
"""


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """Generate the final grounded answer from retrieved chunks."""
    context_block = "\n\n".join(
        f"[{c.source}, page {c.page}]\n{c.text}"
        for c in chunks
    )
    user_prompt = (
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"Answer the question using only the above context, with citations."
    )
    return _call_llm(ANSWER_SYSTEM, user_prompt, max_tokens=1024)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(query: str, collection: chromadb.Collection) -> PipelineResult:
    """
    Full self-correcting RAG pipeline.
    Returns a PipelineResult with the decision, answer, and audit trail.
    """
    log: List[str] = []

    # ── A. Retrieve ───────────────────────────────────────────────────────
    log.append(f"[A] Retrieving cross-source chunks for: '{query}'")
    chunks = retrieve_cross_source(query, collection)
    if not chunks:
        log.append("[A] No chunks returned from vector store.")
        return PipelineResult(
            decision=DecisionType.INSUFFICIENT_CLARIFY,
            answer="I couldn't find any relevant information in the knowledge base. Could you clarify what document or topic you're asking about?",
            chunks_used=[],
            judge_result=None,
            log_entries=log,
        )

    top_score = chunks[0].score
    log.append(f"[A] Top similarity score: {top_score:.4f} (threshold={CONFIDENCE_THRESHOLD})")

    # ── B. Confidence gate ────────────────────────────────────────────────
    sufficient_by_score, _ = check_confidence(chunks)

    if not sufficient_by_score:
        log.append(f"[B] Score below threshold — attempting query reformulation.")

        reformed = reformulate_query(query)
        log.append(f"[B] Reformulated query: '{reformed}'")

        rechunks = retrieve_cross_source(reformed, collection)
        re_sufficient, re_top = check_confidence(rechunks)
        log.append(f"[B] Re-query top score: {re_top:.4f}")

        if re_sufficient:
            chunks = rechunks
            log.append("[B] Re-query improved results — proceeding with reformulated chunks.")
            sufficient_by_score = True
        else:
            log.append("[B] Re-query still insufficient — returning clarification request.")
            return PipelineResult(
                decision=DecisionType.INSUFFICIENT_CLARIFY,
                answer=(
                    "I don't have enough grounded information to answer this confidently. "
                    "The knowledge base doesn't seem to contain relevant material for this question. "
                    f"Could you clarify: **{query}** — specifically what document, date range, or context you're asking about?"
                ),
                chunks_used=rechunks,
                judge_result=None,
                requery_attempted=True,
                reformulated_query=reformed,
                confidence_score=re_top,
                log_entries=log,
            )

    # ── C. Contradiction / sufficiency judge ──────────────────────────────
    log.append("[C] Running LLM judge for sufficiency + contradiction check.")
    judge = judge_context(query, chunks)
    log.append(f"[C] Judge → sufficient={judge.sufficient}, contradictory={judge.contradictory}")
    log.append(f"[C] Reasoning: {judge.reasoning}")

    # ── D. Branch on judge output ─────────────────────────────────────────

    # Case 1: Contradiction detected
    if judge.contradictory:
        log.append("[D] Branch: CONTRADICTORY — surfacing conflict to user.")
        conflict_lines = ""
        if judge.conflicting_claims:
            conflict_lines = "\n".join(
                f"- {claim}" for claim in judge.conflicting_claims
            )
        else:
            # Build conflict summary from top sources
            sources = list({c.source for c in chunks[:3]})
            conflict_lines = f"Sources involved: {', '.join(sources)}"

        answer = (
            "⚠️ **The sources in the knowledge base disagree on this topic.**\n\n"
            f"**Conflicting claims found:**\n{conflict_lines}\n\n"
            f"**Judge's reasoning:** {judge.reasoning}\n\n"
            "Which perspective are you asking about, or would you like both sides summarised?"
        )
        return PipelineResult(
            decision=DecisionType.CONTRADICTORY,
            answer=answer,
            chunks_used=chunks,
            judge_result=judge,
            confidence_score=top_score,
            log_entries=log,
        )

    # Case 2: Judge says insufficient (even though score was OK)
    if not judge.sufficient:
        log.append("[D] Branch: Judge says insufficient — attempting re-query.")
        reformed = reformulate_query(query)
        log.append(f"[D] Reformulated query: '{reformed}'")
        rechunks = retrieve_cross_source(reformed, collection)
        re_judge = judge_context(query, rechunks)
        log.append(f"[D] Re-judge → sufficient={re_judge.sufficient}")

        if re_judge.sufficient and not re_judge.contradictory:
            chunks = rechunks
            judge = re_judge
            log.append("[D] Re-query successful — proceeding to answer generation.")
        else:
            log.append("[D] Still insufficient after re-query — returning clarification request.")
            return PipelineResult(
                decision=DecisionType.INSUFFICIENT_CLARIFY,
                answer=(
                    "I don't have enough grounded information in the knowledge base to answer this question confidently.\n\n"
                    f"**What I can tell you:** {judge.reasoning}\n\n"
                    "Could you clarify what specific aspect you're looking for, or provide additional documents?"
                ),
                chunks_used=chunks,
                judge_result=judge,
                requery_attempted=True,
                reformulated_query=reformed,
                confidence_score=top_score,
                log_entries=log,
            )

    # Case 3: Sufficient & consistent → generate grounded answer
    log.append("[D] Branch: SUFFICIENT + CONSISTENT — generating grounded answer.")
    answer = generate_answer(query, chunks)
    log.append("[E] Answer generated successfully.")

    return PipelineResult(
        decision=DecisionType.GROUNDED_ANSWER,
        answer=answer,
        chunks_used=chunks,
        judge_result=judge,
        confidence_score=top_score,
        log_entries=log,
    )


# ── Naive baseline (no self-correction) ──────────────────────────────────────
def run_naive_pipeline(query: str, collection: chromadb.Collection) -> str:
    """Naive RAG: retrieve top-k and always answer, no confidence gate."""
    chunks = retrieve(query, collection, k=TOP_K)
    if not chunks:
        return "No relevant documents found."
    return generate_answer(query, chunks)
