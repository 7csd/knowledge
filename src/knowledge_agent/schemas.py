from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(StrictModel):
    doc_id: str = Field(min_length=1, max_length=80)
    version_date: date
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    scope: str = "未说明"
    status: str = "未说明"
    content: str = Field(min_length=1)


class SourceLocation(StrictModel):
    file_id: str
    source_path: str
    sha256: str
    method: Literal["markdown", "pdf_text", "ocr", "pdf_table"]
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    line_start: int | None = None
    line_end: int | None = None
    confidence: float | None = None


class Passage(StrictModel):
    doc_id: str
    version_date: str
    chunk_id: str
    title: str
    topic: str
    scope: str
    status: str
    start: int
    end: int
    text: str
    parent_id: str | None = None
    related_ids: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    index_text: str | None = None
    token_count: int | None = None
    kind: str = "text"
    table_id: str | None = None
    row_id: str | None = None
    quality_warnings: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.doc_id}@{self.version_date}#{self.chunk_id}"


class Citation(StrictModel):
    doc_id: str
    version_date: str
    chunk_id: str
    quote: str = Field(min_length=1, max_length=2000)
    source_locations: list[SourceLocation] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.doc_id}@{self.version_date}#{self.chunk_id}"


class Answer(StrictModel):
    status: Literal["completed", "needs_clarification", "insufficient_evidence", "tool_failed", "incomplete"]
    answer: str = Field(min_length=1, max_length=8000)
    citations: list[Citation] = Field(default_factory=list, max_length=30)
    gaps: list[str] = Field(default_factory=list, max_length=30)
    conflicts: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def require_evidence_or_gap(self):
        if self.status == "completed" and not self.citations:
            raise ValueError("completed 必须提供原文引用")
        if self.status != "completed" and not self.gaps:
            raise ValueError("非 completed 状态必须说明资料缺口或未完成原因")
        if self.status == "completed" and self.conflicts:
            raise ValueError("conflicts表示未解决冲突，不能同时返回completed")
        return self


class HistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class RunResult(StrictModel):
    run_id: str
    model_mode: str
    retrieval_mode: str
    model_steps: int
    tool_calls: int
    result: Answer
