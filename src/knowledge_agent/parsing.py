"""Layout-aware source extraction. OCR is computed from page pixels, never sidecar text."""

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

from .schemas import SourceLocation

PARSER_VERSION = "layout-v3"
HEADING = re.compile(r"^第[一二三四五六七八九十百零〇\d]+[条节章]")
NOTE = re.compile(r"^注\s*\d+\s*[：:]")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def bounds(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def parser_identity():
    names = ["pdfplumber", "pypdfium2", "rapidocr-onnxruntime", "onnxruntime"]
    return {"version": PARSER_VERSION, "packages": {n: importlib.metadata.version(n) for n in names},
            "ocr_dpi": 180, "ocr_det_limit_side_len": 1536, "ocr_text_score": 0.5}


class SourceParser:
    def __init__(self, cache: Path):
        self.cache = cache
        self.identity = parser_identity()
        self.ocr = None

    def location(self, entry, path, text, method, **kwargs):
        try:
            source_path = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            source_path = str(path.resolve())
        return SourceLocation(file_id=entry["file_id"], source_path=source_path,
                              sha256=entry["sha256"], method=method, text=text, **kwargs).model_dump()

    def parse(self, entry: dict, path: Path) -> dict:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"{entry['file_id']}文件哈希与清单不一致，请先更新清单")
        key = hashlib.sha256(json.dumps({"file": actual, "parser": self.identity,
                                        "doc_id": entry["doc_id"], "version_date": entry["version_date"],
                                        "title": entry["title"]},
                                       sort_keys=True).encode()).hexdigest()
        cache_path = self.cache / key / "parsed.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            # Paths are relocated on cache reuse, so snapshots remain usable from another checkout.
            for block in result["blocks"]:
                for loc in block["sources"]:
                    loc.update(self.location(entry, path, loc["text"], loc["method"],
                                             **{k: v for k, v in loc.items() if k not in {
                                                 "file_id", "source_path", "sha256", "text", "method"}}))
            return result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.markdown(entry, path) if path.suffix.lower() == ".md" else self.pdf(entry, path, cache_path.parent)
        result["parser"] = self.identity
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def markdown(self, entry, path):
        raw = path.read_text(encoding="utf-8-sig")
        if entry["doc_id"] not in raw or entry["version_date"] not in raw:
            raise ValueError("Markdown正文缺少文档ID或版本日期")
        blocks, heading = [], entry["title"]
        lines = raw.splitlines()
        buffer, start = [], 0

        def flush(end):
            if buffer:
                text = "\n".join(buffer)
                blocks.append({"kind": "text", "heading": heading, "text": text,
                               "sources": [self.location(entry, path, text, "markdown",
                                                         line_start=start, line_end=end)]})
                buffer.clear()

        for n, line in enumerate(lines, 1):
            if line.startswith("#"):
                flush(n - 1)
                heading = line.lstrip("# ")
            elif (not line.strip() or line.startswith(">") or re.match(
                r"^- (文档 ID|版本日期|主题|适用范围|状态)：", line
            )):
                flush(n - 1)
            else:
                if not buffer:
                    start = n
                buffer.append(line)
        flush(len(lines))
        return {"blocks": blocks, "pages": [], "warnings": []}

    def ocr_lines(self, page, renderer, n, cache, file_hash):
        ocr_identity = {k: v for k, v in self.identity.items() if k != "version"}
        ocr_key = hashlib.sha256(json.dumps([file_hash, ocr_identity], sort_keys=True).encode()).hexdigest()
        raw_cache = self.cache / "ocr" / ocr_key
        raw_cache.mkdir(parents=True, exist_ok=True)
        raw_path = raw_cache / f"page-{n + 1}.json"
        if raw_path.exists():
            if not (cache / f"page-{n + 1}.png").exists():
                bitmap = renderer[n].render(scale=180 / 72)
                bitmap.to_pil().save(cache / f"page-{n + 1}.png")
                bitmap.close()
            return json.loads(raw_path.read_text(encoding="utf-8"))
        if self.ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                                det_limit_side_len=1536, text_score=0.5)
        bitmap = renderer[n].render(scale=180 / 72)
        image = bitmap.to_pil().convert("RGB")
        image.save(cache / f"page-{n + 1}.png")
        import numpy as np

        result, _ = self.ocr(np.asarray(image))
        lines = []
        for box, text, confidence in result or []:
            xs, ys = [p[0] / 2.5 for p in box], [p[1] / 2.5 for p in box]
            lines.append({"text": text, "x0": min(xs), "top": min(ys),
                          "x1": max(xs), "bottom": max(ys), "confidence": float(confidence)})
        lines.sort(key=lambda ln: (round(ln["top"] / 5), ln["x0"]))
        raw_path.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
        bitmap.close()
        return lines

    def pdf(self, entry, path, cache):
        import pdfplumber
        import pypdfium2

        blocks, pages, warnings = [], [], []
        current = None
        last_table = None
        table_counter = 0
        table_rows = {}
        full_metadata = []

        def flush_current():
            nonlocal current
            if current:
                current["text"] = "\n\n".join(s["text"] for s in current["sources"])
                blocks.append(current)
                current = None

        with pdfplumber.open(path) as pdf:
            renderer = pypdfium2.PdfDocument(str(path))
            try:
                for n, page in enumerate(pdf.pages, 1):
                    native = (page.extract_text() or "").strip()
                    # Bad/replacement text also requires OCR; a text layer alone is insufficient.
                    use_ocr = len(native) < 60 or native.count("\ufffd") > len(native) * 0.01
                    lines = self.ocr_lines(page, renderer, n - 1, cache, entry["sha256"]) if use_ocr else page.extract_text_lines()
                    if not lines:
                        raise ValueError(f"{entry['file_id']}第{n}页没有可用文字，停止发布索引")
                    full_metadata.extend(l["text"] for l in lines if l["top"] < 135)
                    method = "ocr" if use_ocr else "pdf_text"
                    tables = [] if use_ocr else page.find_tables()
                    pages.append({"page": n, "method": method, "line_count": len(lines),
                                  "table_count": len(tables), "image_cache": str(cache / f"page-{n}.png") if use_ocr else None,
                                  "min_confidence": min(l["confidence"] for l in lines) if use_ocr else None})
                    if use_ocr and any(re.match(r"T\d{2}", compact(l["text"])) for l in lines):
                        raise ValueError("扫描表格需额外表格识别，当前不能把OCR行冒充结构化表格")
                    page_table_ids = []
                    for table in tables:
                        data = table.extract()
                        if not data or len(data) < 2:
                            continue
                        header = [compact(v or "") for v in data[0]]
                        table_title = next((compact(l["text"]) for l in lines
                                            if re.search(r"表\s*[A-Za-z一二三\d]+", l["text"])
                                            and l["top"] < table.bbox[1]), "")
                        title_key = re.search(r"表([A-Za-z一二三\d]+)", table_title)
                        title_key = title_key.group(0) if title_key else table_title
                        signature = (tuple(header), title_key)
                        if last_table and last_table["page"] == n - 1 and last_table["signature"] == signature:
                            table_id = last_table["id"]
                        else:
                            table_counter += 1
                            table_id = f"table{table_counter:02}"
                        last_table = {"page": n, "signature": signature, "id": table_id}
                        page_table_ids.append(table_id)
                        for i, values in enumerate(data[1:], 1):
                            values = [compact(v or "") for v in values]
                            if not any(values):
                                continue
                            row_id = values[0]
                            base_id = re.sub(r"[（(]续[）)]", "", row_id)
                            sources = [self.location(entry, path, value, "pdf_table", page=n,
                                                     bbox=box) for value, box in zip(values, table.rows[i].cells)
                                       if value and box]
                            key = (table_id, base_id)
                            continuation = base_id != row_id
                            if continuation:
                                previous = table_rows.get(key)
                                if (not previous or previous["sources"][-1]["page"] != n - 1
                                        or list(table_rows)[-1] != key):
                                    raise ValueError("表格续行缺少可靠的相邻行依据")
                                previous["sources"].extend(sources)
                                previous["text"] += "\n" + " | ".join(values)
                                previous["index_body"] += "；续行说明：" + values[-1]
                            else:
                                if key in table_rows:
                                    raise ValueError("同表存在重复行号，无法可靠定位")
                                row = {"kind": "table_row", "heading": title_key, "table_id": table_id,
                                       "row_id": row_id, "text": " | ".join(values), "sources": sources,
                                       "index_body": "；".join(f"{col}：{v}" for col, v in zip(header, values))}
                                table_rows[key] = row
                                blocks.append(row)
                    paragraph = []

                    def flush_paragraph(paragraph=paragraph, use_ocr=use_ocr, method=method, n=n):
                        nonlocal current
                        if not paragraph:
                            return
                        text = "".join(l["text"] for l in paragraph)
                        box = bounds([[l["x0"], l["top"], l["x1"], l["bottom"]] for l in paragraph])
                        conf = min(l["confidence"] for l in paragraph) if use_ocr else None
                        source = self.location(entry, path, text, method, page=n, bbox=box, confidence=conf)
                        if current is None:
                            current = {"kind": "text", "heading": entry["title"], "sources": []}
                        current["sources"].append(source)
                        paragraph.clear()

                    for line in lines:
                        text = line["text"].strip()
                        if line["top"] < 135 or line["bottom"] > page.height - 50:
                            continue
                        if any(t.bbox[1] <= (line["top"] + line["bottom"]) / 2 <= t.bbox[3] for t in tables):
                            continue
                        if re.match(r"^\d{2}\s*[\u4e00-\u9fff]", text) or text.startswith("续表见下页"):
                            continue
                        if HEADING.match(compact(text)) or NOTE.match(compact(text)) or text.startswith("计价单位"):
                            flush_paragraph()
                            flush_current()
                            kind = "table_note" if NOTE.match(compact(text)) else (
                                "table_caption" if text.startswith("计价单位") else "text")
                            current = {"kind": kind, "heading": text, "sources": []}
                            if kind.startswith("table_"):
                                current["table_id"] = page_table_ids[0] if page_table_ids else None
                            paragraph.append(line)
                            if kind == "text":
                                flush_paragraph()  # preserve the heading as original text too
                        else:
                            if paragraph and line["top"] - paragraph[-1]["bottom"] > 12:
                                flush_paragraph()
                            paragraph.append(line)
                    flush_paragraph()
                    # A clause can continue across pages; each page keeps separate coordinates.
                    if current and current["kind"].startswith("table_"):
                        flush_current()
                    if not tables:
                        last_table = None
                flush_current()
            finally:
                renderer.close()
        metadata_text = compact("".join(full_metadata))
        if compact(entry["doc_id"]) not in metadata_text or entry["version_date"] not in metadata_text:
            raise ValueError("PDF实际解析结果缺少清单中的文档ID或版本日期，请核对OCR与清单")
        unique, captions = [], {}
        for block in blocks:
            if block["kind"] == "table_caption":
                key = (block.get("table_id"), block["text"])
                if key in captions:
                    captions[key]["sources"].extend(block["sources"])
                    continue
                captions[key] = block
            unique.append(block)
        blocks = unique
        for block in blocks:
            block["warnings"] = []
            for source in block["sources"]:
                score = source["confidence"]
                if score is not None and score < 0.8:
                    block["warnings"].append("blocked: OCR置信度低于0.8，需复核原页")
                elif score is not None and score < 0.95:
                    block["warnings"].append("OCR置信度低于0.95，数字、日期及否定词需核对原页")
        return {"blocks": blocks, "pages": pages, "warnings": warnings}
