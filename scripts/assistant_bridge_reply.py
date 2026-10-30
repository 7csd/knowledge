"""Utilities for writing one externally chosen decision after inspecting a request."""
import json
from pathlib import Path

ROOT = Path("examples/assistant-evaluation/2026-10-30")


def reply(cid, step, summary, calls):
    request = ROOT / cid / f"decision-{step:02d}.request.json"
    if not request.exists():
        raise ValueError(f"No observed request: {request}")
    response = request.with_name(request.name.replace(".request.", ".response."))
    value = {"role": "assistant", "content": summary, "tool_calls": [
        {"id": f"{cid}-{step}-{i}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        for i, (name, args) in enumerate(calls)]}
    with response.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def search(query):
    return "search_documents", {"query": query}


def read(key):
    doc, rest = key.split("@")
    date, chunk = rest.split("#")
    return "read_passage", {"doc_id": doc, "version_date": date, "chunk_id": chunk}


def related(doc, date):
    return "list_related_documents", {"doc_id": doc, "version_date": date}


def submit(answer, status="completed", citations=(), gaps=(), conflicts=()):
    return "submit_result", {"answer": answer, "status": status, "citations": list(citations),
                                 "gaps": list(gaps), "conflicts": list(conflicts)}


def citation(key, quote):
    return {**read(key)[1], "quote": quote}
