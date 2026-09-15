from pathlib import Path

import pytest

from bastion.benchmarks.profiling import read_collapsed, summarize_profile

HANDLER = "main (uvicorn/main.py:1);score (bastion/serving/app.py:103)"
FEATURES = "features_from_snapshot (bastion/features/online.py:280)"
WINDOWS = "window_aggregates_many (bastion/features/definitions.py:90)"

PROFILE = f"""\
{HANDLER};{FEATURES};{WINDOWS} 6
{HANDLER};{FEATURES};window_aggregates (bastion/features/definitions.py:120);{WINDOWS} 2
{HANDLER};execute (redis/asyncio/client.py:1500);parse_response (redis/asyncio/client.py:700) 3

<module> (bastion script:10);run (bastion/serving/decision_log.py:95) 1
"""


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "profile.txt"
    path.write_text(PROFILE)
    return path


def test_frames_with_spaces_keep_their_count(tmp_path: Path) -> None:
    stacks = read_collapsed(_profile(tmp_path))
    assert len(stacks) == 4  # the blank line is skipped
    frames, count = stacks[-1]
    assert frames == ["<module> (bastion script:10)", "run (bastion/serving/decision_log.py:95)"]
    assert count == 1


def test_stages_count_each_stack_once_even_when_nested(tmp_path: Path) -> None:
    summary = summarize_profile(_profile(tmp_path))
    assert summary.samples == 12
    assert summary.stage_samples == {
        "score handler": 11,
        "Redis pipeline": 3,
        "features_from_snapshot": 8,
        "window aggregates": 8,  # the nested wrapper + implementation stack counts once
        "distinct counts": 0,
        "LightGBM predict": 0,
        "decision log writer": 1,
    }
    assert summary.share("features_from_snapshot") == pytest.approx(8 / 12)
    assert summary.leaves[0] == (WINDOWS, 8)


def test_busy_time_per_request_uses_the_sampling_interval(tmp_path: Path) -> None:
    summary = summarize_profile(_profile(tmp_path))
    # 8 samples at 250 Hz are 32 ms of busy time, spread over 16 requests.
    assert summary.ms_per_request("window aggregates", rate_hz=250, requests=16) == 2.0


def test_empty_profile_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no samples"):
        summarize_profile(path)
