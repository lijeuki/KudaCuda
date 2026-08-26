import unittest
import json
import tempfile
from pathlib import Path

from KudaCuda.tools.benchmark.analyze_promotion_gate import analyze_promotion_gate, read_samples


class PromotionGateTests(unittest.TestCase):
    def test_promotes_when_candidate_p25_beats_control_p75(self):
        result = analyze_promotion_gate(
            candidate_values=[30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            control_values=[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        )

        self.assertTrue(result["promoted"])
        self.assertGreater(result["candidate_p25"], result["control_p75"])

    def test_rejects_when_candidate_p25_does_not_beat_control_p75(self):
        result = analyze_promotion_gate(
            candidate_values=[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            control_values=[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        )

        self.assertFalse(result["promoted"])
        self.assertEqual(result["reason"], "candidate_p25_not_above_control_p75")

    def test_rejects_missing_samples_without_inventing_percentiles(self):
        result = analyze_promotion_gate(candidate_values=[], control_values=[1.0])

        self.assertFalse(result["promoted"])
        self.assertIsNone(result["candidate_p25"])
        self.assertEqual(result["reason"], "missing_samples")

    def test_reads_named_samples_from_aggregate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(
                json.dumps(
                    {
                        "candidate_static_graph": {
                            "steady_state_tokens_per_second": [50.0, 52.0]
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                read_samples(path, section="candidate_static_graph"),
                [50.0, 52.0],
            )


if __name__ == "__main__":
    unittest.main()
