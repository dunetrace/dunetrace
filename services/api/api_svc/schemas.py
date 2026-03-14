"""
Response models for the API. Uses Pydantic v2 in production, falls back to
stdlib dataclasses so tests can run without it installed.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel as _Base

    class _Model(_Base):
        pass

    _PYDANTIC = True
except ImportError:
    from dataclasses import dataclass, field as _field
    _PYDANTIC = False
    _Base = object  # type: ignore

    # Minimal compat shim: dataclass with a model_dump() method
    class _Model:  # type: ignore
        def model_dump(self):
            import dataclasses
            return dataclasses.asdict(self)


if _PYDANTIC:
    # Pydantic models (production)

    class Page(_Model):
        total:    int
        offset:   int
        limit:    int
        has_more: bool

    class AgentSummary(_Model):
        agent_id:       str
        last_seen:      Optional[float]
        run_count:      int
        signal_count:   int
        critical_count: int
        high_count:     int
        sparkline:      List[int] = []              # 7 daily signal counts, oldest→newest
        failure_types:  Dict[str, int] = {}         # { "TOOL_LOOP": 3, ... }

    class AgentListResponse(_Model):
        agents: List[AgentSummary]
        page:   Page

    class RunSummary(_Model):
        run_id:        str
        agent_id:      str
        agent_version: str
        started_at:    Optional[float]
        completed_at:  Optional[float]
        exit_reason:   Optional[str]
        step_count:    int
        signal_count:  int
        has_signals:   bool

    class RunEvent(_Model):
        event_type:    str
        step_index:    int
        timestamp:     float
        payload:       Dict[str, Any]
        parent_run_id: Optional[str]

    class RunSignal(_Model):
        id:               int
        failure_type:     str
        severity:         str
        step_index:       int
        confidence:       float
        detected_at:      float
        evidence:         Dict[str, Any]
        title:            str
        what:             str
        why_it_matters:   str
        evidence_summary: str
        suggested_fixes:  List[Dict[str, Any]]

    class RunDetail(_Model):
        run_id:        str
        agent_id:      str
        agent_version: str
        started_at:    Optional[float]
        completed_at:  Optional[float]
        exit_reason:   Optional[str]
        step_count:    int
        events:        List[RunEvent]
        signals:       List[RunSignal]

    class RunListResponse(_Model):
        runs: List[RunSummary]
        page: Page

    class SignalDetail(_Model):
        id:               int
        failure_type:     str
        severity:         str
        run_id:           str
        agent_id:         str
        agent_version:    str
        step_index:       int
        confidence:       float
        detected_at:      float
        evidence:         Dict[str, Any]
        alerted:          bool
        title:            str
        what:             str
        why_it_matters:   str
        evidence_summary: str
        suggested_fixes:  List[Dict[str, Any]]

    class SignalListResponse(_Model):
        signals: List[SignalDetail]
        page:    Page

    class HealthResponse(_Model):
        status:  str = "ok"
        version: str = "0.1.0"
        db:      str = "unknown"

    # Insights models

    class InputHashPattern(_Model):
        input_hash:      str
        failure_type:    str
        triggered_count: int
        total_runs:      int
        rate:            float

    class VersionStat(_Model):
        agent_version:     str
        run_count:         int
        runs_with_signals: int
        signal_count:      int
        signal_rate:       float
        first_seen:        Optional[float]
        last_seen:         Optional[float]

    class SignalTrendPoint(_Model):
        failure_type:  str
        agent_version: str
        day:           str
        count:         int

    class TimeToFirstTool(_Model):
        p25:            Optional[float]
        p50:            Optional[float]
        p75:            Optional[float]
        avg_steps:      Optional[float]
        runs_with_tool: int
        total_runs:     int
        daily_trend:    List[Dict[str, Any]]

    class HourlyPatternPoint(_Model):
        hour_of_day:  int
        run_count:    int
        signal_count: int
        signal_rate:  float

    class AgentInsights(_Model):
        input_patterns: List[InputHashPattern]
        signal_trends:  List[SignalTrendPoint]
        versions:       List[VersionStat]
        time_to_tool:   TimeToFirstTool
        hourly_pattern: List[HourlyPatternPoint]

else:
    # Stdlib dataclass fallback (sandbox / testing)
    from dataclasses import dataclass, field

    @dataclass
    class Page:
        total: int
        offset: int
        limit: int
        has_more: bool
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class AgentSummary:
        agent_id: str
        last_seen: Optional[float]
        run_count: int
        signal_count: int
        critical_count: int
        high_count: int
        sparkline: List[int] = _field(default_factory=list)
        failure_types: Dict[str, int] = _field(default_factory=dict)
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class AgentListResponse:
        agents: List[Any]
        page: Page
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class RunSummary:
        run_id: str; agent_id: str; agent_version: str
        started_at: Optional[float]; completed_at: Optional[float]
        exit_reason: Optional[str]; step_count: int
        signal_count: int; has_signals: bool
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class RunEvent:
        event_type: str; step_index: int; timestamp: float
        payload: Dict[str, Any]; parent_run_id: Optional[str]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class RunSignal:
        id: int; failure_type: str; severity: str
        step_index: int; confidence: float; detected_at: float
        evidence: Dict[str, Any]; title: str; what: str
        why_it_matters: str; evidence_summary: str
        suggested_fixes: List[Dict[str, Any]]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class RunDetail:
        run_id: str; agent_id: str; agent_version: str
        started_at: Optional[float]; completed_at: Optional[float]
        exit_reason: Optional[str]; step_count: int
        events: List[Any]; signals: List[Any]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class RunListResponse:
        runs: List[Any]; page: Page
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class SignalDetail:
        id: int; failure_type: str; severity: str
        run_id: str; agent_id: str; agent_version: str
        step_index: int; confidence: float; detected_at: float
        evidence: Dict[str, Any]; alerted: bool
        title: str; what: str; why_it_matters: str
        evidence_summary: str; suggested_fixes: List[Dict[str, Any]]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class SignalListResponse:
        signals: List[Any]; page: Page
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class HealthResponse:
        status: str = "ok"
        version: str = "0.1.0"
        db: str = "unknown"
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)
