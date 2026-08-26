import json
import tempfile
import unittest
from pathlib import Path

from KudaCuda.harness.compare import compare_values, write_json_evidence
from KudaCuda.tools.benchmark.benchmark_gemma import _should_check_eos, summarize_generation
from KudaCuda.tools.profiling.profile_command import build_profile_command


class ComparisonTests(unittest.TestCase):
    def test_exact_integer_comparison_passes(self):
        result = compare_values([1, 2, 3], [1, 2, 3], exact=True)

        self.assertTrue(result["passed"])
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_tolerance_comparison_records_error_metrics(self):
        result = compare_values([1.0, 2.001], [1.0, 2.0], atol=0.01, rtol=0.0)

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["max_abs_error"], 0.001, places=6)
        self.assertEqual(result["nan_actual"], 0)
        self.assertEqual(result["inf_mismatch"], 0)

    def test_tolerance_comparison_rejects_large_error(self):
        result = compare_values([1.0, 2.2], [1.0, 2.0], atol=0.01, rtol=0.0)

        self.assertFalse(result["passed"])
        self.assertGreater(result["max_rel_error"], 0.0)

    def test_comparison_counts_nan_and_inf_mismatches(self):
        result = compare_values([float("nan"), float("inf")], [0.0, 1.0])

        self.assertFalse(result["passed"])
        self.assertEqual(result["nan_actual"], 1)
        self.assertEqual(result["inf_mismatch"], 1)

    def test_evidence_writer_creates_sorted_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "evidence.json"
            write_json_evidence(path, {"z": 1, "a": "ok"})

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"a": "ok", "z": 1})


class ProfilingTests(unittest.TestCase):
    def test_profile_command_accepts_resolved_profiler_path(self):
        command = build_profile_command(
            "nsys",
            ["python", "-m", "demo"],
            "results/nsys-demo",
            executable="C:/NVIDIA/nsys.exe",
        )

        self.assertEqual(
            command,
            [
                "C:/NVIDIA/nsys.exe",
                "profile",
                "--force-overwrite=true",
                "-o",
                "results/nsys-demo",
                "python",
                "-m",
                "demo",
            ],
        )


class GemmaBenchmarkTests(unittest.TestCase):
    def test_deferred_stop_checks_are_opt_in(self):
        self.assertTrue(_should_check_eos(False))
        self.assertFalse(_should_check_eos(True))

    def test_generation_summary_does_not_invent_ttft_or_throughput(self):
        result = summarize_generation(
            prefill_ms=12.0,
            decode_ms=[4.0, 6.0],
            generated_tokens=3,
            peak_vram_mib=2048.0,
        )

        self.assertEqual(result["ttft_ms"], 12.0)
        self.assertEqual(result["itl_ms"], 5.0)
        self.assertEqual(result["steady_state_tokens_per_second"], 200.0)
        self.assertAlmostEqual(result["total_ms"], 22.0)
        self.assertAlmostEqual(result["tokens_per_second"], 3 / 0.022)
        self.assertEqual(result["peak_vram_mib"], 2048.0)

    def test_generation_summary_marks_missing_decode_itl(self):
        result = summarize_generation(
            prefill_ms=12.0,
            decode_ms=[],
            generated_tokens=1,
            peak_vram_mib=None,
        )

        self.assertIsNone(result["itl_ms"])
        self.assertIsNone(result["steady_state_tokens_per_second"])
        self.assertIsNone(result["peak_vram_mib"])


if __name__ == "__main__":
    unittest.main()
