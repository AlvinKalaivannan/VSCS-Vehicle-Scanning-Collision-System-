"""Regenerate ``docs/REPORT.md`` from ``metrics/results.jsonl``.

CLAUDE.md section 7.4: the report shows the latest value per metric, the trend per
metric, and pass/fail against the acceptance thresholds from section 6 (which live in
``configs/eval.yaml``).

Two rules shape this module:

* **Nothing is invented.** Every number in the report comes from a line in
  ``results.jsonl``. An acceptance gate with no measurement is reported as *not
  measured*, never as passing (CLAUDE.md sections 0.4 and 9).
* **An empty metrics file is a valid input.** At the start of the project there are
  no measurements, and ``python scripts/report.py`` must still succeed - that is the
  P0-T2 acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vscs.common.config import load_config, repo_root
from vscs.common.io import git_commit_info, metrics_path, read_jsonl

REQUIRED_FIELDS = ("ts", "task", "metric", "value")


@dataclass
class MetricSummary:
    """Everything the report needs to say about one metric."""

    metric: str
    task: str | None
    latest_value: float | None
    latest_ts: str | None
    latest_git: str | None
    latest_run_dir: str | None
    latest_split: str | None
    n_samples: int
    previous_value: float | None
    limit: float | None
    direction: str | None

    # -- derived ----------------------------------------------------------- #
    @property
    def status(self) -> str:
        """``pass`` / ``fail`` / ``no gate`` / ``not measured``."""
        if self.latest_value is None:
            return "not measured"
        if self.limit is None or self.direction is None:
            return "no gate"
        if self.direction == "below":
            return "pass" if self.latest_value < self.limit else "fail"
        return "pass" if self.latest_value > self.limit else "fail"

    @property
    def trend(self) -> str:
        """Movement since the previous measurement, in the direction that is good."""
        if self.latest_value is None or self.previous_value is None:
            return "-"
        delta = self.latest_value - self.previous_value
        if abs(delta) < 1e-12:
            return "flat"
        if self.direction == "below":
            better = delta < 0
        elif self.direction == "above":
            better = delta > 0
        else:
            return f"{delta:+.4g}"
        arrow = "improved" if better else "worse"
        return f"{arrow} ({delta:+.4g})"


def load_metrics(path: Path | None = None) -> list[dict[str, Any]]:
    """Read ``results.jsonl``, skipping nothing and validating each line.

    A missing file is treated as "no metrics yet" rather than an error, so the report
    can be generated on a fresh clone.
    """
    p = Path(path) if path is not None else metrics_path()
    if not p.is_file():
        return []
    records = list(read_jsonl(p))
    for i, rec in enumerate(records, start=1):
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise ValueError(f"{p}: line {i} is missing required field(s) {missing}: {rec}")
    return records


def summarize(
    records: list[dict[str, Any]],
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> list[MetricSummary]:
    """Collapse the append-only log into one summary per metric.

    Ordering within a metric is by ``ts``, so "latest" means most recently recorded,
    not last in the file. Metrics that have a threshold but no measurement are
    included with ``latest_value=None`` so an unmet gate cannot silently vanish from
    the report.
    """
    thresholds = thresholds or {}
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_metric.setdefault(str(rec["metric"]), []).append(rec)

    summaries: list[MetricSummary] = []
    for metric in sorted(set(by_metric) | set(thresholds)):
        rows = sorted(by_metric.get(metric, []), key=lambda r: str(r["ts"]))
        spec = thresholds.get(metric, {})
        latest = rows[-1] if rows else None
        previous = rows[-2] if len(rows) >= 2 else None
        summaries.append(
            MetricSummary(
                metric=metric,
                task=str(latest["task"]) if latest else spec.get("task"),
                latest_value=float(latest["value"]) if latest else None,
                latest_ts=str(latest["ts"]) if latest else None,
                latest_git=str(latest.get("git", "")) if latest else None,
                latest_run_dir=str(latest.get("run_dir", "")) if latest else None,
                latest_split=str(latest.get("split", "")) if latest else None,
                n_samples=len(rows),
                previous_value=float(previous["value"]) if previous else None,
                limit=spec.get("limit"),
                direction=spec.get("direction"),
            )
        )
    return summaries


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _gate(s: MetricSummary) -> str:
    if s.limit is None or s.direction is None:
        return "-"
    return f"{'<' if s.direction == 'below' else '>'} {s.limit:g}"


_STATUS_MARK = {
    "pass": "PASS",
    "fail": "FAIL",
    "no gate": "-",
    "not measured": "not measured",
}


def render_markdown(summaries: list[MetricSummary], *, n_records: int) -> str:
    """Render the report body. Auto-generated - never hand-edited."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    git = git_commit_info()
    measured = [s for s in summaries if s.latest_value is not None]
    unmeasured = [s for s in summaries if s.latest_value is None]
    failing = [s for s in measured if s.status == "fail"]

    lines: list[str] = [
        "# VSCS metrics report",
        "",
        "<!-- AUTO-GENERATED by scripts/report.py. Do not edit by hand: your changes",
        "     will be overwritten. Edit the metrics by appending to",
        "     metrics/results.jsonl, or change the gates in configs/eval.yaml. -->",
        "",
        f"Generated {now} at commit `{git['short'] or 'nogit'}`"
        f"{' (working tree dirty)' if git['dirty'] == 'true' else ''}.",
        "",
        f"Source: `metrics/results.jsonl`, {n_records} "
        f"measurement{'' if n_records == 1 else 's'} recorded.",
        "",
    ]

    if not measured:
        lines += [
            "## No metrics measured yet",
            "",
            "`metrics/results.jsonl` holds no measurements, so this report has no numbers",
            "in it. That is the correct state before any evaluation has run - every number",
            "in VSCS has to trace to a line in that file (CLAUDE.md sections 7.4 and 9),",
            "and none may be written by hand.",
            "",
        ]
    else:
        lines += [
            "## Summary",
            "",
            f"- {len(measured)} metric{'' if len(measured) == 1 else 's'} measured",
            f"- {len(failing)} failing its acceptance gate",
            f"- {len(unmeasured)} acceptance gate"
            f"{'' if len(unmeasured) == 1 else 's'} with no measurement yet",
            "",
            "## Measured metrics",
            "",
            "| Metric | Task | Latest | Gate | Status | Trend | n | Split | Evidence |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in sorted(measured, key=lambda x: (x.status != "fail", x.metric)):
            run = f"`{s.latest_run_dir}`" if s.latest_run_dir else "-"
            lines.append(
                f"| `{s.metric}` | {s.task or '-'} | {_fmt(s.latest_value)} | {_gate(s)} "
                f"| {_STATUS_MARK[s.status]} | {s.trend} | {s.n_samples} "
                f"| {s.latest_split or '-'} | {run} |"
            )
        lines.append("")

    if failing:
        lines += [
            "## Failing gates",
            "",
            "These do not meet the acceptance criteria in CLAUDE.md section 6.",
            "",
        ]
        for s in failing:
            lines.append(
                f"- **`{s.metric}`** ({s.task or 'no task'}): {_fmt(s.latest_value)}, "
                f"needs {_gate(s)}"
            )
        lines.append("")

    if unmeasured:
        lines += [
            "## Acceptance gates with no measurement yet",
            "",
            "Declared in `configs/eval.yaml` but never measured. Not passing, not failing -",
            "simply unproven.",
            "",
            "| Metric | Task | Gate |",
            "|---|---|---|",
        ]
        for s in sorted(unmeasured, key=lambda x: (x.task or "", x.metric)):
            lines.append(f"| `{s.metric}` | {s.task or '-'} | {_gate(s)} |")
        lines.append("")

    lines += [
        "---",
        "",
        "Regenerate with `python scripts/report.py`.",
        "",
    ]
    return "\n".join(lines)


def default_report_path() -> Path:
    return repo_root() / "docs" / "REPORT.md"


def generate_report(
    *,
    metrics_file: Path | None = None,
    out_path: Path | None = None,
    eval_config: dict[str, Any] | None = None,
) -> Path:
    """Write ``docs/REPORT.md`` and return its path. Safe on an empty metrics file."""
    records = load_metrics(metrics_file)
    cfg = eval_config if eval_config is not None else load_config("eval")
    thresholds = cfg.get("thresholds") or {}
    summaries = summarize(records, thresholds)

    out = Path(out_path) if out_path is not None else default_report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summaries, n_records=len(records)), encoding="utf-8")
    return out
