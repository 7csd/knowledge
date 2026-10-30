"""Reproducible structure-first child indexing with separately readable parent evidence."""

import json
import re
import sys
from pathlib import Path

from .incremental import classify_file, load_base, source_record
from .parsing import SourceParser
from .schemas import Document, Passage


class TokenChunker:
    def __init__(self, tokenizer, target=300, max_input=450, overlap=40, parent_limit=1000):
        if not 0 <= overlap < target or not 32 <= target < max_input <= 512 or parent_limit < target:
            raise ValueError("切分参数不合法")
        self.tokenizer = tokenizer
        self.target, self.max_input, self.overlap, self.parent_limit = target, max_input, overlap, parent_limit

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def windows(self, text: str, limit: int, overlap=0):
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            if self.count(text[start:low]) > limit:
                raise ValueError("token预算不足以容纳一个字符")
            while low < high:
                middle = (low + high + 1) // 2
                if self.count(text[start:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            end = low
            if end < len(text):
                boundaries = [m.end() + start for m in re.finditer(r"[。！？；\n]", text[start:end])]
                boundaries = [v for v in boundaries if v >= start + (end - start) * 0.6]
                if boundaries:
                    end = boundaries[-1]
            yield start, end
            if end == len(text):
                break
            next_start = end
            if overlap:
                while next_start > start + 1 and self.count(text[next_start - 1:end]) <= overlap:
                    next_start -= 1
            start = next_start


def select_files(manifest, scan_docs):
    files = [m for m in manifest["files"] if m.get("canonical")]
    available = {m["doc_id"] for m in files}
    if not set(scan_docs) <= available:
        raise ValueError("指定的扫描对照文档不存在")
    for doc_id in scan_docs:
        choices = [m for m in manifest["files"] if m["doc_id"] == doc_id
                   and m["representation"] == "image_only_pdf"]
        if len(choices) != 1:
            raise ValueError("扫描对照文件必须唯一")
        files = [choices[0] if m["doc_id"] == doc_id else m for m in files]
    return files


def ingest(manifest_path: Path, output: Path, cache: Path, tokenizer_model: str,
           revision: str, local_only=False, scan_docs=(), target=300, overlap=40,
           base: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=False)
    parser = SourceParser(cache)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, revision=revision, local_files_only=local_only,
                                            trust_remote_code=False)
    chunker = TokenChunker(tokenizer, target=target, overlap=overlap)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = select_files(manifest, scan_docs)
    config = {"strategy": "structure-parent-child-v1", "target_tokens": target,
              "max_input_tokens": 450, "long_clause_overlap": overlap, "parent_limit": 1000,
              "tokenizer": tokenizer_model, "tokenizer_revision": revision, "parser": parser.identity,
              "selected_file_hashes": {m["file_id"]: m["sha256"] for m in entries}}
    documents, parents, children, records = load_base(base, config)
    if base:
        from .store import DocumentStore
        config["base_fingerprint"] = DocumentStore.load(base).fingerprint
    reports = []
    try:
        for entry in entries:
            path = (manifest_path.parent / entry["path"]).resolve()
            if not path.is_relative_to((manifest_path.parent / "corpus").resolve()):
                raise ValueError("清单中的正文路径必须位于corpus目录内")
            record = source_record(entry, path)
            duplicate_of = classify_file(record, records)
            if duplicate_of is not None:
                if not any(r["file_id"] == record["file_id"] and r["source_path"] == record["source_path"]
                           for r in records):
                    records.append({**record, "duplicate_of": duplicate_of})
                reports.append({"file_id": entry["file_id"], "md5": record["md5"],
                                "action": "skipped_duplicate", "duplicate_of": duplicate_of})
                continue
            print(f"解析 {entry['file_id']} ({entry['representation']})", file=sys.stderr, flush=True)
            parsed = parser.parse(entry, path)
            blocks = parsed["blocks"]
            if not blocks:
                raise ValueError("文档未解析出正文")
            # Keep logical table continuations together; retain all individual source locations.
            blocks.sort(key=lambda b: (b["sources"][0].get("page") or 0,
                                       (b["sources"][0].get("bbox") or [0, 0])[1]))
            content, doc_parents = "", []
            for block in blocks:
                text = block["text"]
                for a, z in chunker.windows(text, chunker.parent_limit):
                    part = text[a:z]
                    if content:
                        content += "\n\n"
                    start = len(content)
                    content += part
                    parent = Passage(
                        **{k: entry[k] for k in ["doc_id", "version_date", "title", "topic", "scope", "status"]},
                        chunk_id=f"r{len(doc_parents) + 1:03}", start=start, end=len(content), text=part,
                        heading_path=[entry["title"], block["heading"]], source_locations=block["sources"],
                        kind=block["kind"], table_id=block.get("table_id"), row_id=block.get("row_id"),
                        quality_warnings=block.get("warnings", []),
                    )
                    # table index bodies are labels + values, never used as raw quote evidence.
                    parent.index_text = block.get("index_body")
                    doc_parents.append(parent)
            for p in doc_parents:
                if p.kind == "table_row":
                    captions = [q for q in doc_parents if q.table_id == p.table_id and q.kind == "table_caption"]
                    notes = [q for q in doc_parents if q.table_id == p.table_id and q.kind == "table_note"]
                    p.related_ids = [q.chunk_id for q in captions[:1] + notes]
                    if captions:
                        # Carry the original unit into the vector text, without repeated page navigation.
                        unit = captions[0].text.split("。")[0] + "。"
                        p.index_text = (p.index_text or p.text) + "\n" + unit
            doc_children = []
            for parent in doc_parents:
                prefix = (f"{entry['title']} / {parent.heading_path[-1]}\n"
                          f"适用范围：{entry['scope']}\n")
                body = parent.index_text or parent.text
                # A table row should remain complete. Shared notes stay separate and are expanded on read.
                if parent.kind == "table_row" and chunker.count(prefix + body) > chunker.max_input:
                    raise ValueError("表格逻辑行超过Embedding预算，需进一步结构化，不能截掉条件")
                if parent.kind == "table_row":
                    spans = [(0, len(parent.text))]
                else:
                    budget = min(chunker.target, chunker.max_input - chunker.count(prefix) - 3)
                    spans = list(chunker.windows(parent.text, budget, chunker.overlap))
                for a, z in spans:
                    text = parent.text[a:z]
                    index_text = prefix + (body if parent.kind == "table_row" else text)
                    count = chunker.count(index_text)
                    if count > chunker.max_input:
                        raise ValueError("完整Embedding输入超过450 token，停止发布")
                    child = parent.model_copy(update={"chunk_id": f"c{len(doc_children) + 1:03}",
                                                      "parent_id": parent.chunk_id,
                                                      "start": parent.start + a, "end": parent.start + z,
                                                      "text": text, "index_text": index_text, "token_count": count})
                    doc_children.append(child)
                # Parent.index_text is an indexing-only representation; expose original sources on read.
                parent.index_text = None
            doc = Document(**{k: entry[k] for k in ["doc_id", "version_date", "title", "topic", "scope", "status"]},
                           content=content)
            documents.append(doc.model_dump(mode="json"))
            parents.extend(p.model_dump() for p in doc_parents)
            children.extend(p.model_dump() for p in doc_children)
            records.append(record)
            reports.append({"file_id": entry["file_id"], "sha256": entry["sha256"],
                            "md5": record["md5"], "action": "added",
                            "pages": parsed["pages"], "parent_count": len(doc_parents),
                            "child_count": len(doc_children), "table_rows": sum(p.kind == "table_row" for p in doc_parents),
                            "ocr_warnings": sum(bool(p.quality_warnings) for p in doc_parents)})
        snapshot = {"schema_version": 2, "config": config, "documents": documents,
                    "parents": parents, "passages": children, "files": records}
        staged = output / "corpus.pending.json"
        staged.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        from .store import DocumentStore

        DocumentStore.load(staged).verify_sources()
        staged.rename(output / "corpus.json")
        report = {"status": "completed", "scope": "source_parsing_and_chunking_not_llm_acceptance",
                  "documents": len(documents), "parents": len(parents), "children": len(children),
                  "max_embedding_tokens": max(p["token_count"] for p in children),
                  "config": config, "files": reports, "live_model_evaluated": False}
        report["base"] = str(base) if base else None
        report["added_files"] = sum(r["action"] == "added" for r in reports)
        report["skipped_files"] = sum(r["action"] == "skipped_duplicate" for r in reports)
    except Exception as exc:
        (output / "ingestion-report.json").write_text(json.dumps(
            {"status": "failed", "error": str(exc), "completed_files": reports}, ensure_ascii=False, indent=2
        ), encoding="utf-8")
        raise
    (output / "ingestion-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
