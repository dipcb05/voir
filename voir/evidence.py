"""Fail-closed evidence admission and clinical comparability for VOIR."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EvidenceUnit:
    modality: str
    claim: str
    polarity: int
    confidence: float
    uncertainty: float
    patient_id: str
    acquired_at: str
    provenance: Dict[str, Any]
    lesion_id: Optional[str] = None
    specimen_id: Optional[str] = None
    quality_ok: bool = True
    permitted: bool = True
    derivation_id: Optional[str] = None


@dataclass
class DecisionRecord:
    patient_id: str
    index_time: str
    admitted: List[Dict[str, Any]] = field(default_factory=list)
    withheld: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    action: str = "DEFER"
    reason_codes: List[str] = field(default_factory=list)
    bundle_version: str = "unversioned"

    def digest(self) -> str:
        return sha256(repr(asdict(self)).encode("utf-8")).hexdigest()


class EvidenceContract:
    """Validates source records before any learned cross-modal interaction."""
    REQUIRED_PROVENANCE = {"source_id", "source_version", "checksum"}

    def __init__(self, index_time: str, required_modalities: Iterable[str] = ()):
        self.index_time = datetime.fromisoformat(index_time)
        self.required_modalities = set(required_modalities)

    def qualify(self, unit: EvidenceUnit) -> Tuple[bool, Optional[str]]:
        if not unit.quality_ok:
            return False, "QUALITY_FAILURE"
        if not unit.permitted:
            return False, "OUT_OF_SCOPE"
        if not self.REQUIRED_PROVENANCE.issubset(unit.provenance):
            return False, "MISSING_PROVENANCE"
        try:
            if datetime.fromisoformat(unit.acquired_at) > self.index_time:
                return False, "POST_INDEX_EVIDENCE"
        except ValueError:
            return False, "INVALID_ACQUISITION_TIME"
        if unit.polarity not in (-1, 1) or not 0 <= unit.confidence <= 1 or not 0 <= unit.uncertainty <= 1:
            return False, "INVALID_PROPOSITION"
        return True, None

    def admit(self, units: Iterable[EvidenceUnit], record: DecisionRecord) -> List[EvidenceUnit]:
        admitted: List[EvidenceUnit] = []
        seen = set()
        for unit in units:
            ok, reason = self.qualify(unit)
            if not ok:
                record.withheld.append({"modality": unit.modality, "reason": reason})
                continue
            if unit.modality in seen:
                record.withheld.append({"modality": unit.modality, "reason": "DUPLICATE_SOURCE"})
                continue
            seen.add(unit.modality)
            admitted.append(unit)
            record.admitted.append({"modality": unit.modality, "claim": unit.claim, "provenance": unit.provenance})
        record.gaps.extend(sorted(self.required_modalities - seen))
        return admitted

    @staticmethod
    def comparable(a: EvidenceUnit, b: EvidenceUnit) -> bool:
        """Hard patient/lesion/specimen constraint; derivatives are not independent."""
        if a.patient_id != b.patient_id or a.modality == b.modality:
            return False
        if a.lesion_id and b.lesion_id and a.lesion_id != b.lesion_id:
            return False
        if a.specimen_id and b.specimen_id and a.specimen_id != b.specimen_id:
            return False
        return not (a.derivation_id and a.derivation_id == b.derivation_id)
