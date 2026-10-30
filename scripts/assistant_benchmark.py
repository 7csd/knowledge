"""Interactive assistant-in-the-loop evaluation. Never a live API or scripted-model score.

start runs the unchanged LangGraph agent; each decision blocks on a new response file.
pending exposes only questions/history and observations (never gold expectations).
The human/assistant must inspect observations before writing the next response.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from knowledge_agent.agent import KnowledgeAgent
from knowledge_agent.answer_evaluation import audit_run, load_trace, review_template
from knowledge_agent.config import Settings
from knowledge_agent.model import ModelError, ModelMessage
from knowledge_agent.retrieval import BGEEncoder, HybridRetriever
from knowledge_agent.schemas import HistoryMessage
from knowledge_agent.store import DocumentStore
from knowledge_agent.tools import KnowledgeTools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class AssistantBridge:
    mode = "assistant_in_the_loop"

    def __init__(self, directory):
        self.directory, self.step, self.wait_ms = directory, 0, 0.0

    def respond(self, messages, tools):
        self.step += 1
        stem = self.directory / f"decision-{self.step:02d}"
        save(stem.with_suffix(".request.json"), {"messages": messages, "tools": tools})
        started = time.perf_counter()
        response = stem.with_suffix(".response.json")
        while not response.exists():
            if time.perf_counter() - started > 3600:
                raise ModelError("assistant_bridge_timeout")
            time.sleep(0.1)
        raw = json.loads(response.read_text(encoding="utf-8"))
        self.wait_ms += (time.perf_counter() - started) * 1000
        return ModelMessage.model_validate(raw)


class MeasuredTools:
    def __init__(self, tools, case):
        self.tools, self.store, self.retriever = tools, tools.store, tools.retriever
        self.case, self.timings, self.failures = case, [], 0

    def invoke(self, name, arguments, timeout):
        started = time.perf_counter()
        inject = self.case.get("fail_tool") == name and (
            not self.case.get("fail_once") or self.failures == 0)
        if inject:
            self.failures += 1
            obs = {"ok": False, "error": "injected_tool_failure", "message": "显式测试故障注入"}
        else:
            obs = self.tools.invoke(name, arguments, timeout)
        self.timings.append({"name": name, "elapsed_ms": (time.perf_counter()-started)*1000,
                             "injected": inject, "ok": obs.get("ok", False)})
        return obs


def start(root, cases, data_path=None):
    root.mkdir(parents=True, exist_ok=False)
    settings = Settings.from_env()
    if data_path:
        settings.data_path = data_path
    encoder = BGEEncoder(settings.embedding_model, settings.embedding_revision, True)
    encoder.encode(["预热"], query=True)
    store = DocumentStore.load(settings.data_path)
    retriever = HybridRetriever(store, encoder, settings.cache_dir, settings.vector_min_score,
                                settings.vector_db_path)
    tools = KnowledgeTools(store, retriever)
    save(root / "environment.json", {"mode": "assistant_in_the_loop", "corpus": str(settings.data_path),
         "corpus_hash": store.fingerprint, "documents": len(store.documents), "passages": len(store.passages),
         "embedding": encoder.identity, "backend": "qdrant", "api_usage": None,
         "limitations": ["当前助手逐轮代行决策，不是项目HTTP模型验收", "同一助手编题、回答及审阅，有自评偏差",
                         "问题token为BGE tokenizer实测，非对话模型计费token",
                         "wall时间含文件桥接等待，不能作为API推理性能"]})
    lock = threading.Lock()

    def run(case):
        directory = root / case["case_id"]
        directory.mkdir()
        save(directory / "input.json", case)
        model, measured = AssistantBridge(directory), MeasuredTools(tools, case)
        active_settings = settings.model_copy(update={k: case[k] for k in (
            "max_model_steps", "max_tool_calls") if k in case})
        started = time.perf_counter()
        actual = KnowledgeAgent(model, measured, active_settings).run(
            case["question"], [HistoryMessage(**h) for h in case.get("history", [])], directory / "trace.jsonl")
        wall_ms = (time.perf_counter() - started)*1000
        raw = actual.model_dump(mode="json")
        events = load_trace(directory / "trace.jsonl")
        save(directory / "actual.json", raw)
        save(directory / "audit.json", audit_run(raw, events, case))
        save(directory / "review.json", review_template(raw))
        with lock:
            tokenizer = encoder.model.tokenizer
            question_tokens = len(tokenizer.encode(case["question"], add_special_tokens=False))
            embedding_tokens = len(tokenizer.encode(
                "为这个句子生成表示以用于检索相关文章：" + case["question"], add_special_tokens=True))
        save(directory / "telemetry.json", {"question_bge_tokens": question_tokens,
             "question_embedding_input_tokens": embedding_tokens, "llm_prompt_tokens": None,
             "llm_completion_tokens": None, "llm_total_tokens": None, "llm_api_latency_ms": None,
             "wall_ms_including_assistant_wait": wall_ms, "assistant_bridge_wait_ms": model.wait_ms,
             "business_tools": measured.timings, "business_tool_ms": sum(t["elapsed_ms"] for t in measured.timings)})
        print(case["case_id"], actual.result.status, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            list(pool.map(run, cases))
    finally:
        tools.close()


def pending(root):
    for request in sorted(root.glob("*/decision-*.request.json")):
        if request.with_name(request.name.replace(".request.", ".response.")).exists():
            continue
        messages = json.loads(request.read_text(encoding="utf-8"))["messages"]
        observations = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            obs = json.loads(m["content"])
            if "hits" in obs:
                obs["hits"] = [{"key": f"{h['doc_id']}@{h['version_date']}#{h['chunk_id']}",
                                "snippet": h["snippet"]} for h in obs["hits"]]
            if "passage" in obs:
                obs = {"ok": obs["ok"], "read": [{"key": f"{p['doc_id']}@{p['version_date']}#{p['chunk_id']}",
                       "text": p["text"]} for p in [obs["passage"], *obs.get("related_passages", [])]]}
            observations.append(obs)
        print(json.dumps({"case": request.parent.name, "step": request.name.split('.')[0],
                          "input": [m for m in messages if m["role"] in ("user",)][:2],
                          "observations": list(reversed(observations))}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "pending"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("data/assistant_evaluation_cases.json"))
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.command == "start":
        start(args.output, json.loads(args.cases.read_text(encoding="utf-8")), args.data)
    else:
        pending(args.output)
