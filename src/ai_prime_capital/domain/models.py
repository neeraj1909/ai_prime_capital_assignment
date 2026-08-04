from dataclasses import dataclass
from math import isfinite


def _require_non_blank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    

@dataclass(frozen=True, slots=True)
class SourceLocator:
    page_number: int | None = None
    section: str | None = None
    paragraph_index: int | None = None
    table_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    
    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive")
        
        for name, value in (
            ("paragraph_index", self.paragraph_index),
            ("table_index", self.table_index),
            ("start_offset", self.start_offset),
            ("end_offset", self.end_offset),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
            
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must be provided together")
        
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset < self.start_offset
        ):
            raise ValueError("end_offset must not precede start_offset")
        
        if self.section is not None:
            _require_non_blank(self.section, "section")
            
            
@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    text: str
    locator: SourceLocator
    
    def __post_init__(self) -> None:
        _require_non_blank(self.text, "text")
        
    
@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str 
    file_name: str 
    text: str
    locator: SourceLocator
    
    def __post_init__(self) -> None:
        _require_non_blank(self.chunk_id, "chunk_id")
        _require_non_blank(self.document_id, "document_id")
        _require_non_blank(self.file_name, "file_name")
        _require_non_blank(self.text, "text")
        

@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    chunk: Chunk
    score: float
    
    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        
        
@dataclass(frozen=True, slots=True)
class ModelDecision:
    # Intentionally permits invalid combinations. The grounding policy must
    # reject malformed or unsupported model output instead of trusting it.
    answerable: bool
    answer: str = ""
    citations: tuple[str, ...] = ()
    refused: bool = False
    
    
@dataclass(frozen=True, slots=True)
class SourceReference:
    document_id: str
    file_name: str
    chunk_id: str
    snippet: str
    score: float
    locator: SourceLocator
    
    
@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str
    grounded: bool
    sources: tuple[SourceReference, ...]
    
    def __post_init__(self) -> None:
        _require_non_blank(self.answer, "answer")
        if self.grounded != bool(self.sources):
            raise ValueError("grounded must match whether sources are present")
