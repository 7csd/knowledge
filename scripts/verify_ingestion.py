"""Actual multi-format/OCR/BGE/Qdrant verification. No live LLM claims."""

import argparse
import difflib
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from knowledge_agent.cli import build_tools
from knowledge_agent.config import Settings
from knowledge_agent.ingestion import ingest
from knowledge_agent.offline import run_demo
from knowledge_agent.retrieval import HybridRetriever
from knowledge_agent.store import DocumentStore

ROOT = Path(__file__).resolve().parents[1]


def verify_source_change(encoder, settings):
    with tempfile.TemporaryDirectory(prefix="source-change-", dir=ROOT / "tmp") as temporary:
        root = Path(temporary)
        corpus = root / "corpus"
        corpus.mkdir()
        manifest = json.loads((ROOT / "data/source_materials/manifest.json").read_text(encoding="utf-8"))
        entry = next(m for m in manifest["files"] if m["doc_id"] == "D03")
        original = (ROOT / "data/source_materials" / entry["path"]).read_text(encoding="utf-8")
        source = corpus / "annual.md"
        manifest_path = root / "manifest.json"
        stores, indexes, observations = [], [], []
        try:
            stale_rejected = False
            for number in [3, 5]:
                source.write_text(original.replace("提前3个工作日", f"提前{number}个工作日"), encoding="utf-8")
                if stores:
                    try:
                        stores[0].verify_sources()
                    except ValueError:
                        stale_rejected = True
                entry = {**entry, "path": "corpus/annual.md",
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                manifest_path.write_text(json.dumps({"files": [entry]}, ensure_ascii=False), encoding="utf-8")
                out = root / f"snapshot-{number}"
                ingest(manifest_path, out, settings.cache_dir / "parsed", settings.embedding_model,
                       settings.embedding_revision, True)
                store = DocumentStore.load(out / "corpus.json")
                index = HybridRetriever(store, encoder, settings.cache_dir, settings.vector_min_score,
                                        root / f"database-{number}")
                stores.append(store)
                indexes.append(index)
                hits = index.search("年假提前多久申请")
                evidence = [p for hit in hits for p in store.read_bundle(
                    hit["doc_id"], hit["version_date"], hit["chunk_id"])]
                assert any(f"提前{number}个工作日" in p.text for p in evidence)
                observations.append({"source_value": number, "cache_key": index.cache_key,
                                     "hits": hits, "read_texts": [p.text for p in evidence]})
            assert indexes[0].cache_key != indexes[1].cache_key and stale_rejected
            assert not any("提前3个工作日" in p.text for p in stores[1].parents.values())
            return {"source_hash_change_rejected_stale_snapshot": stale_rejected,
                    "new_index_identity": indexes[0].cache_key != indexes[1].cache_key,
                    "runs": observations, "scope": "real_source_reparse_and_reindex_not_llm_answer_change"}
        finally:
            for index in indexes:
                index.close()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/corpus.json"))
    parser.add_argument("--scan-data", type=Path, default=Path("examples/ingestion/scan-variant-final/corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("examples/ingestion/verification"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings(data_path=args.data,
                        embedding_revision="7999e1d3359715c523056ef9478215996d62a620",
                        embedding_local_only=True)
    tools = build_tools(settings)
    report = {"scope": "real_ocr_embedding_database_and_offline_protocol_only",
              "live_model_evaluated": False, "corpus_hash": tools.store.fingerprint}
    try:
        store, index = tools.store, tools.retriever
        encoder = index.encoder
        counts = [len(encoder.model.tokenizer.encode(p.index_text, add_special_tokens=True))
                  for p in store.passages.values()]
        assert max(counts) <= 450
        assert all("人民币元" in p.index_text for p in store.passages.values() if p.kind == "table_row")
        assert len(store.documents) == 12
        report["index"] = {"documents": len(store.documents), "parents": len(store.parents),
                           "children": len(store.passages), "max_embedding_tokens": max(counts),
                           "database_path": str(index.database.path),
                           "database_points": index.database.client.count(index.database.collection).count}
        assert report["index"]["database_points"] == len(store.passages)
        cases = [
            ("annual", "申请年假需要提前多久", [("D03", "提前3个工作日")]),
            ("conflict", "报销期限多久", [("D01", "报销期限为费用发生后30个自然日"),
                                          ("D02", "报销期限为费用发生后15个自然日")]),
            ("paraphrase", "我垫付的钱多久以内得找公司报账", [("D01", "报销期限为费用发生后30个自然日"),
                                                            ("D02", "报销期限为费用发生后15个自然日")]),
            ("vpn", "VPN连接失败怎么办", [("D05", "先检查网络与验证码有效性")]),
            ("export", "普通成员可以把星云数据导出CSV吗", [("D06", "仅空间管理员具有导出权限")]),
            ("password", "现行企业账号密码最少多少位", [("D09", "至少12位")]),
            ("table", "成都出差会展期间酒店涨到500元可以直接报吗", [("D04", "350"),
                                                                    ("D04", "事先取得部门负责人书面批准")]),
            ("outside", "公司买房补贴是多少", []),
        ]
        rows = []
        for case_id, query, requirements in cases:
            hits = index.search(query)
            evidence = {}
            observations = []
            for hit in hits:
                request = {k: hit[k] for k in ["doc_id", "version_date", "chunk_id"]}
                observation = tools.invoke("read_passage", request, 15)
                assert observation["ok"]
                observations.append({"arguments": request, "observation": observation})
                for p in [observation["passage"], *observation["related_passages"]]:
                    evidence[(p["doc_id"], p["chunk_id"])] = p
            found = [any(p["doc_id"] == doc_id and needle in p["text"] for p in evidence.values())
                     for doc_id, needle in requirements]
            row = {"case_id": case_id, "input": query, "hits": hits, "reads": observations,
                   "required_evidence": requirements, "required_found": found,
                   "all_required_found": all(found) if requirements else None,
                   "answer_generated": False}
            if case_id == "paraphrase" and not all(found):
                # Explicit diagnostic query, not a model-generated decision or hidden synonym rule.
                recovery_hits = index.search("报销期限 费用发生后 自然日", topic="报销期限")
                recovery = [p for h in recovery_hits for p in store.read_bundle(
                    h["doc_id"], h["version_date"], h["chunk_id"])]
                row["diagnostic_requery"] = {"query": "报销期限 费用发生后 自然日",
                                             "origin": "developer_authored_not_llm", "hits": recovery_hits,
                                             "all_required_found": all(any(
                                                 p.doc_id == d and needle in p.text for p in recovery
                                             ) for d, needle in requirements)}
            rows.append(row)
        report["retrieval"] = rows
        positives = [r for r in rows if r["required_evidence"]]
        report["retrieval_summary"] = {
            "cases": len(positives), "all_required_found_at_6_with_parent_expansion": sum(
                r["all_required_found"] for r in positives),
            "failed_cases": [r["case_id"] for r in positives if not r["all_required_found"]],
            "note": "Hand-authored development probes; not an independent benchmark or live answer evaluation",
        }
        table = next(p for p in store.passages.values() if p.doc_id == "D04" and p.row_id == "T08")
        bundle = store.read_bundle(table.doc_id, table.version_date, table.chunk_id)
        pages = {loc.page for p in bundle for loc in p.source_locations}
        assert {2, 3, 5} <= pages
        assert len([p for p in store.parents.values() if p.kind == "table_row"]) == 32
        report["cross_page_table"] = {"required_pages_present": True, "pages": sorted(pages),
                                      "logical_rows": 32, "read_bundle": [p.model_dump() for p in bundle]}
        scanned = DocumentStore.load(args.scan_data)
        original_text = store.documents["D05", "2026-01-01"].content
        scan_text = scanned.documents["D05", "2026-01-01"].content
        normalize = lambda s: "".join(s.split())
        key_facts = ["企业账号和动态验证码", "先检查网络与验证码有效性", "不要向他人分享验证码"]
        assert all(k in normalize(scan_text) for k in key_facts)
        report["ocr_comparison"] = {
            "doc_id": "D05", "native_characters": len(original_text), "ocr_characters": len(scan_text),
            "normalized_sequence_similarity": difflib.SequenceMatcher(
                None, normalize(original_text), normalize(scan_text), autojunk=False).ratio(),
            "key_fact_checks": {k: k in normalize(scan_text) for k in key_facts},
            "note": "Similarity to native parser output, not manually annotated OCR character accuracy",
            "ocr_sources": [loc.model_dump() for p in scanned.parents.values() if p.doc_id == "D05"
                            for loc in p.source_locations],
        }
        report["source_change"] = verify_source_change(encoder, settings)
    finally:
        tools.close()
    # Reopen the same on-disk collection, verifying persistence rather than just in-process data.
    reopened = build_tools(settings)
    try:
        report["database_reopen"] = {"count": reopened.retriever.database.client.count(
            reopened.retriever.database.collection).count, "search_hits": reopened.retriever.search("报销期限多久")}
        assert report["database_reopen"]["count"] == report["index"]["children"]
    finally:
        reopened.close()
    report["offline_protocol_runs"] = {}
    expected_status = {"normal": "completed", "conflict": "needs_clarification",
                       "missing": "insufficient_evidence", "failure": "tool_failed"}
    for scenario in ["normal", "conflict", "missing", "failure"]:
        actual = run_demo(scenario, args.output / f"offline-{scenario}", settings)
        report["offline_protocol_runs"][scenario] = actual["actual"]["result"]["status"]
        assert report["offline_protocol_runs"][scenario] == expected_status[scenario]
    report["completed_checks"] = True
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in ["index", "retrieval_summary", "offline_protocol_runs",
                                                            "completed_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
