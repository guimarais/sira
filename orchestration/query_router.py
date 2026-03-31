"""
Module for classifying query intent and dispatching to the appropriate retrievers.
"""
import asyncio
import re

import anthropic

from config import settings
from retrieval.sql_retriever import retrieve_structured
from retrieval.vector_retriever import retrieve_chunks
from utils.prompt_loader import load_prompt

# Keywords that strongly suggest structured stock data queries
_STOCK_KEYWORDS = re.compile(
    r"\b(stock|ticker|price|market\s*cap|p/?e\s*ratio|earnings|eps|dividend|"
    r"target\s*price|shares|valuation|buy|sell|hold|analyst|forecast|revenue|"
    r"profit|loss|quarterly|annual\s*report)\b",
    re.IGNORECASE,
)

# Keywords that strongly suggest unstructured macro/strategic queries
_MACRO_KEYWORDS = re.compile(
    r"\b(macro|inflation|interest\s*rate|gdp|federal\s*reserve|fed|monetary|"
    r"fiscal|recession|economic|economy|sector|industry|geopolit|trade\s*war|"
    r"yield\s*curve|credit|liquidity|central\s*bank|policy|outlook|trend)\b",
    re.IGNORECASE,
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _classify_by_keywords(query: str) -> str | None:
    """Return 'stock', 'macro', 'hybrid', or None if ambiguous."""
    has_stock = bool(_STOCK_KEYWORDS.search(query))
    has_macro = bool(_MACRO_KEYWORDS.search(query))

    if has_stock and has_macro:
        return "hybrid"
    if has_stock:
        return "stock"
    if has_macro:
        return "macro"
    return None


def _classify_by_llm(query: str) -> str:
    """Ask Claude to classify the intent. Returns 'stock', 'macro', or 'hybrid'."""
    try:
        p = load_prompt("query_classification")
        response = _get_client().messages.create(
            model=p["model"],
            max_tokens=p["max_tokens"],
            messages=[{"role": "user", "content": p["user_template"].format(query=query)}],
        )
        label = response.content[0].text.strip().lower()
        return label if label in ("stock", "macro", "hybrid") else "hybrid"
    except anthropic.AuthenticationError:
        raise
    except anthropic.APIConnectionError:
        raise
    except Exception:
        # Any other classification failure: default to hybrid so both retrievers run.
        return "hybrid"


def _safe_retrieve_chunks(query: str) -> list[dict]:
    """Wrap retrieve_chunks so a failure returns an empty list instead of raising."""
    try:
        return retrieve_chunks(query)
    except Exception:
        return []


def _safe_retrieve_structured(query: str) -> dict:
    """Wrap retrieve_structured so a failure returns an error dict instead of raising."""
    try:
        return retrieve_structured(query)
    except Exception as exc:
        return {"sql_generated": "", "rows": [], "error": str(exc)}


async def route_query(query: str) -> dict:
    """Classify query intent and dispatch to the appropriate retriever(s).

    Uses keyword heuristics first; falls back to Claude for ambiguous queries.
    For 'hybrid' queries, both retrievers run in parallel. A failure in one
    retriever does not abort the other.

    Args:
        query: The natural-language investment research question.

    Returns:
        dict with keys:
            intent (str): 'stock', 'macro', or 'hybrid'.
            vector_results (list[dict]): PDF chunk results (empty for stock-only).
            sql_results (dict): SQL query results (empty rows for macro-only).
    """
    intent = _classify_by_keywords(query)
    if intent is None:
        intent = _classify_by_llm(query)

    loop = asyncio.get_event_loop()

    if intent == "hybrid":
        vector_task = loop.run_in_executor(None, _safe_retrieve_chunks, query)
        sql_task = loop.run_in_executor(None, _safe_retrieve_structured, query)
        vector_results, sql_results = await asyncio.gather(vector_task, sql_task)

    elif intent == "stock":
        vector_results = []
        sql_results = await loop.run_in_executor(None, _safe_retrieve_structured, query)

    else:  # macro
        vector_results = await loop.run_in_executor(None, _safe_retrieve_chunks, query)
        sql_results = {"sql_generated": "", "rows": [], "error": None}

    return {
        "intent": intent,
        "vector_results": vector_results,
        "sql_results": sql_results,
    }
