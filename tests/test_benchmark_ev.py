"""Tests: the EV benchmark script runs, its invariants hold, and the committed report matches its template.

The numbers in ``docs/ev_benchmark.md`` are machine-dependent timings and so are not compared; what is
checked is that the script produces every section, that the consistency checks it reports pass, and that
the committed document has the same structure as what the script generates.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("cantools")
pytest.importorskip("httpx")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
import benchmark_ev_drive as bench  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return bench.run_all(quick=True)


def _headings(text: str) -> list[str]:
    return re.findall(r"^#{1,3} .*$", text, flags=re.M)


class TestQuickRun:
    def test_every_section_is_present(self, results):
        assert set(results) == {"environment", "cycles", "convergence", "legacy", "conditions", "limits",
                                "reproducibility", "can", "gateway_throughput", "gateway_latency",
                                "gateway_replay", "auth"}
        assert results["environment"]["bms"]

    def test_the_energy_account_closes(self, results):
        for c in results["cycles"]:
            assert c["closure_vehicle"] < 1e-8 and c["closure_pack"] < 1e-3 and c["closure_ocv"] < 1e-3
            assert c["trace_rms_kmh"] < 1e-3                      # an unconstrained car follows the trace

    def test_consumption_converges_and_the_pack_residual_falls_with_the_step(self, results):
        conv = results["convergence"]
        assert all(b["pack_residual_Wh"] < a["pack_residual_Wh"] for a, b in zip(conv, conv[1:]))
        assert max(abs(x["vs_finest_pct"]) for x in conv) < 0.5

    def test_the_forward_model_agrees_with_the_inverse_model(self, results):
        assert all(abs(x["difference_pct"]) < 1.0 for x in results["legacy"])

    def test_the_run_is_reproducible_everywhere_it_is_checked(self, results):
        rep = results["reproducibility"]
        checks = {k: v for k, v in rep.items() if isinstance(v, bool)}
        assert len(checks) == 6 and all(checks.values()), checks

    def test_can_output_decodes_within_the_quantization_floor_with_nothing_missing(self, results):
        can = results["can"]
        assert can["decode_ok"] and can["missing"] == 0 and can["received"] == can["frames"]
        assert all(can["max_error"][k] <= can["tolerance"][k] + 1e-9 for k in can["tolerance"])
        assert can["frames_per_s"] > 0

    def test_the_gateway_delivers_everything_and_reports_sane_timings(self, results):
        for t in results["gateway_throughput"]:
            assert t["all_received"] and t["per_client_records_per_s"] > 0
        lat = results["gateway_latency"]
        assert lat["received"] == lat["records"] and 0.0 < lat["p50_ms"] <= lat["p99_ms"] <= lat["max_ms"]
        assert results["gateway_replay"]["received"] == results["gateway_replay"]["records"]
        assert results["auth"]["verify_per_s"] > 0 and results["auth"]["issue_per_s"] > 0


class TestReport:
    def test_the_markdown_is_complete_and_has_no_unfilled_placeholders(self, results):
        text = bench.render_markdown(results)
        for part in ("# Virtual EV benchmark results", "## Standard cycles", "### Does the energy account close?",
                     "## Reproducibility", "## Virtual CAN output", "## Telemetry gateway",
                     "## What these results do not show"):
            assert part in text
        assert "{" not in text and "}" not in text
        assert not re.search(r"\bnan\b|\binf\b", text, flags=re.I)

    def test_the_report_says_up_front_that_it_is_not_validation(self, results):
        text = bench.render_markdown(results)
        assert "not** validation against a physical vehicle" in text.split("## Standard cycles")[0]

    def test_prose_that_states_a_trend_follows_the_data(self, results):
        broken = {**results, "convergence": [{**x, "pack_residual_Wh": 1.0 + i} for i, x in
                                              enumerate(results["convergence"])]}
        assert "does not fall monotonically" in bench.render_markdown(broken)
        limited = {**results, "cycles": [{**c, "trace_rms_kmh": 5.0} for c in results["cycles"]]}
        assert "a limit stops the car keeping up" in bench.render_markdown(limited)

    def test_main_writes_the_report(self, results, tmp_path, monkeypatch):
        monkeypatch.setattr(bench, "run_all", lambda quick=False: results)
        out = tmp_path / "report.md"
        assert bench.main(["--quick", "--write", str(out)]) == 0
        assert out.read_text(encoding="utf-8") == bench.render_markdown(results)

    def test_the_committed_report_has_the_structure_the_script_generates(self, results):
        committed = (_ROOT / "docs" / "ev_benchmark.md").read_text(encoding="utf-8")
        generated = _headings(bench.render_markdown(results))
        have = iter(_headings(committed))
        # every heading the (quick) script produces appears in the committed report, in order
        assert all(h in have for h in generated), "docs/ev_benchmark.md is out of step with the script"
        assert "Sensitivity to vehicle and climate" in committed and "When the pack cannot deliver" in committed

    def test_the_table_helper(self):
        assert bench._table(["a", "b"], [[1, 2], [3, 4]]) == "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
