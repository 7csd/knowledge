import json
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .config import Settings
from .model import ChatModel, ModelError, ModelMessage
from .prompts import SYSTEM_PROMPT
from .schemas import Answer, HistoryMessage, Passage, RunResult
from .tools import KnowledgeTools, tool_definitions, validate_citations
from .trace import Trace


class State(TypedDict):
    messages: list[dict]
    evidence: dict[str, Passage]
    model_steps: int
    tool_calls: int
    result: Answer | None


def unfinished(reason: str) -> Answer:
    return Answer(status="incomplete", answer="本次任务未完成，尚不能给出可靠结论。", gaps=[reason])


class KnowledgeAgent:
    def __init__(self, model: ChatModel, tools: KnowledgeTools, settings: Settings):
        self.model, self.tools, self.settings = model, tools, settings

    def run(
        self, question: str, history: list[HistoryMessage] | None = None, trace_path: Path | None = None
    ) -> RunResult:
        if not question.strip() or len(question) > 4000:
            raise ValueError("问题长度应为 1–4000 个字符")
        history = history or []
        if len(history) > 20 or sum(len(m.content) for m in history) > 40000:
            raise ValueError("历史最多 20 条，合计最多 40000 个字符")
        run_id = uuid4().hex
        trace = Trace(run_id, trace_path, (self.settings.api_key.get_secret_value(),))
        trace.record(
            "run_start",
            model_mode=self.model.mode,
            retrieval_mode=self.tools.retriever.mode,
            question=question,
            history=[m.model_dump() for m in history],
            corpus_hash=self.tools.store.fingerprint,
            model_name=self.settings.model,
            max_model_steps=self.settings.max_model_steps,
            max_tool_calls=self.settings.max_tool_calls,
        )

        def model_node(state: State) -> dict:
            if state["model_steps"] >= self.settings.max_model_steps:
                return {"result": unfinished("达到模型决策步数上限")}
            steps = state["model_steps"] + 1
            trace.record("model_request", step=steps)
            try:
                message = self.model.respond(state["messages"], tool_definitions())
            except ModelError as exc:
                trace.record("model_error", step=steps, code=str(exc))
                return {"model_steps": steps, "result": unfinished(f"模型调用失败：{exc}")}
            raw = message.model_dump(exclude_none=True)
            if not message.tool_calls:
                raw.pop("tool_calls", None)
            # Persist only public, brief action summaries; provider reasoning fields are discarded.
            trace.record(
                "model_action",
                step=steps,
                actions=[c.function.name for c in message.tool_calls],
                summary=(message.content or "")[:400],
            )
            messages = state["messages"] + [raw]
            if not message.tool_calls:
                messages.append(
                    {"role": "user", "content": "请调用工具检索核实，或使用 submit_result 提交结果。"}
                )
            return {"messages": messages, "model_steps": steps}

        def tool_node(state: State) -> dict:
            last = ModelMessage.model_validate(state["messages"][-1])
            messages, evidence = list(state["messages"]), dict(state["evidence"])
            count = state["tool_calls"]
            result = None
            for call in last.tool_calls:
                if count >= self.settings.max_tool_calls:
                    result = unfinished("达到工具调用次数上限")
                    trace.record("budget_exhausted", budget="tool_calls")
                    break
                count += 1
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        raise TypeError("工具参数必须为对象")
                except (ValueError, TypeError):
                    args = None
                trace.record(
                    "tool_request",
                    call_id=call.id,
                    name=name,
                    arguments=args if args is not None else "[invalid JSON]",
                )
                if args is None:
                    observation = {"ok": False, "error": "invalid_json_arguments"}
                elif name == "submit_result":
                    try:
                        if len(last.tool_calls) != 1:
                            raise ValueError("submit_result 必须单独调用")
                        candidate = Answer.model_validate(args)
                        errors = validate_citations(candidate, evidence)
                        if errors:
                            observation = {
                                "ok": False,
                                "error": "citation_validation_failed",
                                "details": errors,
                            }
                        else:
                            for citation in candidate.citations:
                                source = evidence[citation.key]
                                citation.source_locations = [
                                    loc for loc in source.source_locations if citation.quote in loc.text
                                ]
                            result = candidate
                            observation = {"ok": True, "status": result.status}
                    except ValidationError as exc:
                        observation = {
                            "ok": False,
                            "error": "invalid_result_schema",
                            "details": [
                                {"field": list(e["loc"]), "message": e["msg"]}
                                for e in exc.errors(include_input=False, include_context=False)
                            ],
                        }
                    except ValueError as exc:
                        observation = {"ok": False, "error": "invalid_result_call", "message": str(exc)}
                else:
                    observation = self.tools.invoke(name, args, self.settings.tool_timeout)
                    if name == "read_passage" and observation.get("ok"):
                        passage = Passage.model_validate(observation["passage"])
                        evidence[passage.key] = passage
                        for item in observation.get("related_passages", []):
                            related = Passage.model_validate(item)
                            evidence[related.key] = related
                trace.record("tool_result", call_id=call.id, name=name, observation=observation)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(observation, ensure_ascii=False),
                    }
                )
                if result:
                    break
            return {"messages": messages, "evidence": evidence, "tool_calls": count, "result": result}

        def after_model(state: State) -> str:
            if state["result"]:
                return "end"
            if state["messages"][-1].get("tool_calls"):
                return "tools"
            return "model"

        graph = StateGraph(State)
        graph.add_node("model", model_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", after_model, {"end": END, "tools": "tools", "model": "model"})
        graph.add_conditional_edges(
            "tools", lambda s: "end" if s["result"] else "model", {"end": END, "model": "model"}
        )
        state = graph.compile().invoke(
            {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                + [m.model_dump() for m in history]
                + [{"role": "user", "content": question}],
                "evidence": {},
                "model_steps": 0,
                "tool_calls": 0,
                "result": None,
            },
            config={"recursion_limit": 2 * self.settings.max_model_steps + 5},
        )
        output = RunResult(
            run_id=run_id,
            model_mode=self.model.mode,
            retrieval_mode=self.tools.retriever.mode,
            model_steps=state["model_steps"],
            tool_calls=state["tool_calls"],
            result=state["result"],
        )
        trace.record("run_end", output=output.model_dump())
        return output
