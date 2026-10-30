"""File identity and append-only snapshot planning, independent of OCR and models."""
import hashlib
import json
from pathlib import Path

from .store import DocumentStore

METADATA = ("doc_id", "version_date", "title", "topic", "scope", "status")


def source_record(entry, path):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != entry["sha256"]:
        raise ValueError(f"{entry['file_id']}文件哈希与清单不一致，请先更新清单")
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    if entry.get("md5", md5) != md5:
        raise ValueError(f"{entry['file_id']}文件MD5与清单不一致")
    return {**entry, "md5": md5, "source_path": path.as_posix()}


def load_base(path, config):
    if path is None:
        return [], [], [], []
    store = DocumentStore.load(path)
    store.verify_sources()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("增量基线必须为多格式快照")
    for key in ("strategy", "target_tokens", "max_input_tokens", "long_clause_overlap",
                "parent_limit", "tokenizer", "tokenizer_revision", "parser"):
        if raw["config"].get(key) != config[key]:
            raise ValueError(f"增量基线配置不一致：{key}，请显式全量重建")
    records = raw.get("files")
    if records is None:  # Adopt the original schema-v2 snapshots without re-parsing.
        records = []
        for doc in raw["documents"]:
            parent = next(p for p in raw["parents"] if p["doc_id"] == doc["doc_id"]
                          and p["version_date"] == doc["version_date"])
            loc = parent["source_locations"][0]
            records.append(source_record({**{k: doc[k] for k in METADATA},
                                          "file_id": loc["file_id"], "sha256": loc["sha256"]},
                                         Path(loc["source_path"])))
    return raw["documents"], raw["parents"], raw["passages"], records


def classify_file(record, records):
    identity = (record["doc_id"], record["version_date"])
    for old in records:
        if ((old["doc_id"], old["version_date"]) == identity
                and (old["sha256"] != record["sha256"] or any(old[k] != record[k] for k in METADATA))):
            raise ValueError("同文档同版本的内容或元数据已变化，请保存独立源文件并增加版本")
    duplicate = next((r for r in records if r["md5"] == record["md5"]), None)
    if duplicate is not None:
        # MD5 is the dedup key; retain SHA256 as an additional integrity check.
        if duplicate["sha256"] != record["sha256"]:
            raise ValueError("MD5相同但SHA256不同，拒绝自动去重")
        return duplicate.get("duplicate_of", duplicate["file_id"])
    return None
