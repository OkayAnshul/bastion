import json
from pathlib import Path

from bastion.benchmarks.latency import RateResult, rebuild_latency_report, write_latency_report

STAGES = ("redis_ms", "features_ms", "model_ms", "total_ms")
STATUSES = {"200": 495, "503": 5}


def _result(
    rate: int, *, dropped: int = 0, status_counts: dict[str, int] | None = None
) -> RateResult:
    client = {"avg": 2.0, "min": 1.0, "med": 1.5, "p(90)": 2.5, "p(95)": 3.0, "p(99)": 4.0}
    return RateResult(
        target_rps=rate,
        duration_s=30,
        achieved_rps=float(rate),
        requests=rate * 30,
        failed_rate=0.0,
        dropped_iterations=dropped,
        client_ms=client | {"max": 9.0},
        server_ms={stage: {"p50": 1.0, "p95": 2.0, "p99": 3.0, "max": 4.0} for stage in STAGES},
        status_counts=status_counts or {},
    )


def _raw_samples(out_dir: Path, rate: int) -> None:
    """A k6 CSV with 500 requests: 1 in 100 answered 503."""
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = ["metric_name,timestamp,metric_value,status"]
    for i in range(500):
        status = "503" if i % 100 == 0 else "200"
        rows += [f"http_reqs,{i},1,{status}", f"http_req_duration,{i},{1 + (i % 50) / 10},{status}"]
    (raw / f"rate_{rate}.csv").write_text("\n".join(rows) + "\n")


def _write(out_dir: Path, results: list[RateResult]) -> Path:
    return write_latency_report(
        results, out_dir, setup={"Service": "test"}, hardware={"cpu": "test"}, revision="abc1234"
    )


def test_headline_is_the_highest_sustained_rate(tmp_path: Path) -> None:
    for rate in (100, 200):
        _raw_samples(tmp_path, rate)
    text = _write(tmp_path, [_result(100), _result(200, dropped=5)]).read_text()
    assert "## Stage breakdown at 100 requests/s" in text
    assert "| 200 | 200.0 | no | 5 |" in text  # an unsustained rate stays in the table
    assert (tmp_path / "figures" / "latency_histogram.png").stat().st_size > 0


def test_rebuild_keeps_the_recorded_numbers_and_revision(tmp_path: Path) -> None:
    _raw_samples(tmp_path, 100)
    _write(tmp_path, [_result(100, status_counts=STATUSES)])
    recorded = json.loads((tmp_path / "latency.json").read_text())
    (tmp_path / "latency.md").unlink()

    rebuild_latency_report(tmp_path)

    assert json.loads((tmp_path / "latency.json").read_text()) == recorded
    assert "`abc1234`" in (tmp_path / "latency.md").read_text()


def test_rebuild_derives_status_codes_missing_from_older_recordings(tmp_path: Path) -> None:
    _raw_samples(tmp_path, 100)
    _write(tmp_path, [_result(100)])  # recorded before status codes were counted

    rebuild_latency_report(tmp_path)

    recorded = json.loads((tmp_path / "latency.json").read_text())
    assert recorded["results"][0]["status_counts"] == STATUSES
    assert "| 200: 495 · 503: 5 |" in (tmp_path / "latency.md").read_text()
