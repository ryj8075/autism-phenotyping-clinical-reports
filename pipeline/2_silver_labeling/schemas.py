from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

class DomainDef(BaseModel):

    code: str
    id: str
    name_ko: str
    name_en: str
    description: str = ""
    pilot_active: bool = True
    category: str = ""
    subcategory: str = ""

def _sort_key(code: str) -> tuple:

    if not code:
        return ("Z", 999)

    prefix = ""
    num_str = ""
    for ch in code:
        if ch.isdigit():
            num_str += ch
        else:
            prefix += ch
    num = int(num_str) if num_str else 0
    return (prefix, num)

class DomainRegistry:

    FALLBACK_DOMAIN_ID = "other_general"

    def __init__(self, domains: List[DomainDef]):
        self.domains = domains
        self.code_to_id: Dict[str, str] = {d.code: d.id for d in domains}
        self.id_to_code: Dict[str, str] = {d.id: d.code for d in domains}
        self.code_to_def: Dict[str, DomainDef] = {d.code: d for d in domains}
        self.id_to_def: Dict[str, DomainDef] = {d.id: d for d in domains}

    @property
    def fallback_id(self) -> str:

        if self.FALLBACK_DOMAIN_ID in self.id_to_code:
            return self.FALLBACK_DOMAIN_ID

        if "other_unclear" in self.id_to_code:
            return "other_unclear"
        return self.FALLBACK_DOMAIN_ID

    def code_to_id_single(self, code: str) -> Optional[str]:

        return self.code_to_id.get(code)

    def id_to_code_single(self, domain_id: str) -> Optional[str]:

        return self.id_to_code.get(domain_id)

    def codes_to_ids(self, codes: List[str]) -> List[str]:

        result = []
        for c in codes:
            if c in self.code_to_id:
                result.append(self.code_to_id[c])
        return result

    def ids_to_codes(self, ids: List[str]) -> List[str]:

        return [self.id_to_code[i] for i in ids if i in self.id_to_code]

    def summary_table(self) -> str:

        lines = []
        for d in sorted(self.domains, key=lambda x: _sort_key(x.code)):
            lines.append(f"  {d.code:>3}. {d.name_ko} ({d.id})")
        return "\n".join(lines)

class LabelItem(BaseModel):

    domain_id: str
    domain_code: Optional[str] = None
    confidence: float = Field(default=0.0)
    rationale: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:

        return max(0.0, min(1.0, float(v)))

class LLMResponse(BaseModel):

    labels: List[LabelItem] = Field(default_factory=list)

class SentenceLabel(BaseModel):

    report_id: str
    sentence_idx: int
    sentence: str
    labels: List[LabelItem] = Field(default_factory=list)
    raw_responses: Optional[List[LLMResponse]] = None
    model: str = ""
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    seed: Optional[int] = None
    error: Optional[str] = None

class GoldLabel(BaseModel):

    report_id: str
    sentence_idx: int
    sentence: str
    labels: List[Union[int, str]] = Field(default_factory=list)

    @field_validator("labels", mode="before")
    @classmethod
    def coerce_to_list(cls, v):

        if v is None:
            return []
        if isinstance(v, (int, str)):
            if isinstance(v, str) and "," in v:
                return [s.strip() for s in v.split(",") if s.strip()]
            return [v]
        return v

class EvalMetrics(BaseModel):

    domain_id: str
    precision: float
    recall: float
    f1: float
    support: int

class EvalReport(BaseModel):

    per_domain: List[EvalMetrics]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float

    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    subset_accuracy: float = 0.0
    hamming_loss: float = 0.0
    total_sentences: int
    threshold_used: float
    model: str
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
