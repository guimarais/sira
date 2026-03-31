"""
Module for synthesising a cited natural-language answer from retrieval results.
"""
import anthropic

from config import settings
from utils.prompt_loader import load_prompt

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _build_context(
    vector_results: list[dict],
    sql_results: dict,
) -> tuple[str, list[str]]:
    """Format retrieval results into a context block and collect source strings.

    Returns:
        (context_text, sources) where sources is a deduplicated list of
        citation strings ready for the response payload.
    """
    sections: list[str] = []
    sources: list[str] = []

    if vector_results:
        pdf_lines: list[str] = []
        for chunk in vector_results:
            citation = f"[Source: {chunk['source']}, p.{chunk['page']}]"
            pdf_lines.append(f"{citation}\n{chunk['text']}")
            if citation not in sources:
                sources.append(citation)
        sections.append("## PDF Context\n" + "\n\n".join(pdf_lines))

    if sql_results.get("rows"):
        sql = sql_results["sql_generated"]
        rows = sql_results["rows"]
        citation = f"[Source: stocks table, SQL: {sql}]"
        rows_text = "\n".join(str(row) for row in rows)
        sections.append(f"## Structured Data Context\n{citation}\n{rows_text}")
        sources.append(citation)

    context = "\n\n".join(sections) if sections else "No context available."
    return context, sources


def build_response(
    query: str,
    vector_results: list[dict],
    sql_results: dict,
) -> dict:
    """Synthesise a cited answer from PDF and SQL retrieval results via Claude.

    Args:
        query: The original user question.
        vector_results: List of dicts from vector_retriever.retrieve_chunks.
        sql_results: Dict from sql_retriever.retrieve_structured.

    Returns:
        dict with keys:
            answer (str): Natural-language answer with inline citations.
            sources (list[str]): All cited sources referenced in the answer.
    """
    context, sources = _build_context(vector_results, sql_results)
    p = load_prompt("response_synthesis")
    user_content = p["user_template"].format(context=context, query=query)

    response = _get_client().messages.create(
        model=p["model"],
        max_tokens=p["max_tokens"],
        messages=[{"role": "user", "content": user_content}],
    )

    return {"answer": response.content[0].text.strip(), "sources": sources}
