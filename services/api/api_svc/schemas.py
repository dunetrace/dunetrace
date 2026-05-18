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
        total_tokens:  Optional[int]   = None
        cost_usd:      Optional[float] = None
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
        total_tokens:  Optional[int]   = None
        cost_usd:      Optional[float] = None
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
        shadow:           bool
        co_signal_count:  int = 0
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

    class FailureRatePoint(_Model):
        failure_type:  str
        day:           str
        total_runs:    int
        affected_runs: int
        rate:          float

    class SystemicPattern(_Model):
        failure_type:  str
        total_runs:    int
        affected_runs: int
        rate:          float
        first_seen:    Optional[float]
        last_seen:     Optional[float]
        is_systemic:   bool

    class DeployEvent(_Model):
        id:          int
        version:     str
        deployed_at: float
        meta:        Dict[str, Any] = {}

    class CostStat(_Model):
        total_cost_usd:        float
        wasted_cost_usd:       float
        wasted_pct:            float
        cost_by_failure_type:  List[Dict[str, Any]] = []

    class DeployRegression(_Model):
        deploy_id:      int
        version:        str
        deployed_at:    float
        before_runs:    int
        before_signals: int
        before_rate:    float
        after_runs:     int
        after_signals:  int
        after_rate:     float
        delta_rate:     float
        is_regression:  bool

    class UserImpact(_Model):
        failure_type:     str
        affected_users:   int
        total_users:      int
        user_impact_rate: float

    class FixRecord(_Model):
        id:                   int
        signal_id:            int
        run_id:               str
        failure_type:         str
        severity:             str
        agent_version:        str
        fix_type:             str
        applied_via:          str
        langfuse_prompt_name: Optional[str]
        langfuse_version:     Optional[int]
        applied_at:           float
        runs_after:           int
        recurrences_after:    int
        verdict:              str

    class FixListResponse(_Model):
        fixes: List[FixRecord]

    class AgentInsights(_Model):
        input_patterns:     List[InputHashPattern]
        signal_trends:      List[SignalTrendPoint]
        versions:           List[VersionStat]
        time_to_tool:       TimeToFirstTool
        hourly_pattern:     List[HourlyPatternPoint]
        failure_rates:      List[FailureRatePoint]
        systemic_patterns:  List[SystemicPattern]
        deploy_events:      List[DeployEvent]      = []
        cost_stats:         Optional[CostStat]     = None
        deploy_regressions: List[DeployRegression] = []
        user_impact:        List[UserImpact]       = []

    class Issue(_Model):
        id:               int
        agent_id:         str
        failure_type:     str
        status:           str           # open | resolved | reopened
        first_seen:       float
        last_seen:        float
        resolved_at:      Optional[float]
        affected_runs:    int
        clean_runs_since: int

    class IssueListResponse(_Model):
        issues: List[Issue]
        total:  int

    # Failure pattern models

    class FailureOverview(_Model):
        affected_runs:      int
        total_runs:         int
        rate:               float
        avg_confidence:     float
        first_seen:         Optional[float]
        last_seen:          Optional[float]
        severity_breakdown: Dict[str, int]

    class StepDistribution(_Model):
        p25:      Optional[float]
        p50:      Optional[float]
        p75:      Optional[float]
        avg_step: Optional[float]

    class DailyTrendPoint(_Model):
        day:           str
        affected_runs: int
        total_runs:    int
        rate:          float

    class CoOccurring(_Model):
        failure_type: str
        co_count:     int
        co_rate:      float

    class TopRun(_Model):
        run_id:      str
        confidence:  float
        severity:    str
        step_index:  int
        detected_at: Optional[float]
        evidence:    Dict[str, Any]

    class FailurePatternAnalysis(_Model):
        failure_type:        str
        overview:            FailureOverview
        step_distribution:   StepDistribution
        evidence_aggregates: List[Dict[str, Any]]
        daily_trend:         List[DailyTrendPoint]
        co_occurring:        List[CoOccurring]
        top_runs:            List[TopRun]

    class AgentHealthScore(_Model):
        agent_id:       str
        score:          Optional[int]       # None when <3 sample runs
        components:     Dict[str, Any]      # keyed by component name
        sample_runs:    int
        baseline_ready: bool                # False when <30 runs; token/latency components are neutral

    class PatternDay(_Model):
        date:          str
        signal_count:  int
        affected_runs: int

    class PatternRow(_Model):
        failure_type:        str
        days:                List[PatternDay]
        total_occurrences:   int
        total_affected_runs: int
        total_runs:          int
        pct_of_runs:         float
        trending_up:         bool

    class AgentPatterns(_Model):
        agent_id: str
        rows:     List[PatternRow]

    class PatternsResponse(_Model):
        agents: List[AgentPatterns]

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
        evidence: Dict[str, Any]; alerted: bool; shadow: bool
        co_signal_count: int
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

    @dataclass
    class FailureRatePoint:
        failure_type: str; day: str; total_runs: int; affected_runs: int; rate: float
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class SystemicPattern:
        failure_type: str; total_runs: int; affected_runs: int
        rate: float; first_seen: Optional[float]; last_seen: Optional[float]; is_systemic: bool
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class AgentHealthScore:
        agent_id: str; score: Optional[int]; components: Dict[str, Any]; sample_runs: int; baseline_ready: bool
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class PatternDay:
        date: str; signal_count: int; affected_runs: int
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class PatternRow:
        failure_type: str; days: List[Any]
        total_occurrences: int; total_affected_runs: int; total_runs: int
        pct_of_runs: float; trending_up: bool
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class AgentPatterns:
        agent_id: str; rows: List[Any]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class PatternsResponse:
        agents: List[Any]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class CostStat:
        total_cost_usd: float; wasted_cost_usd: float; wasted_pct: float
        cost_by_failure_type: List[Any] = _field(default_factory=list)
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class DeployRegression:
        deploy_id: int; version: str; deployed_at: float
        before_runs: int; before_signals: int; before_rate: float
        after_runs: int; after_signals: int; after_rate: float
        delta_rate: float; is_regression: bool
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class UserImpact:
        failure_type: str; affected_users: int; total_users: int; user_impact_rate: float
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class FixRecord:
        id: int; signal_id: int; run_id: str; failure_type: str; severity: str
        agent_version: str; fix_type: str; applied_via: str
        langfuse_prompt_name: Optional[str]; langfuse_version: Optional[int]
        applied_at: float; runs_after: int; recurrences_after: int; verdict: str
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class FixListResponse:
        fixes: List[Any]
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)

    @dataclass
    class AgentInsights:
        input_patterns: List[Any]; signal_trends: List[Any]; versions: List[Any]
        time_to_tool: Any; hourly_pattern: List[Any]; failure_rates: List[Any]
        systemic_patterns: List[Any]; deploy_events: List[Any] = _field(default_factory=list)
        cost_stats: Optional[Any] = None
        deploy_regressions: List[Any] = _field(default_factory=list)
        user_impact: List[Any] = _field(default_factory=list)
        def model_dump(self): import dataclasses; return dataclasses.asdict(self)
