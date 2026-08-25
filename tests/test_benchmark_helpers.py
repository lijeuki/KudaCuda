import unittest

from tools.benchmark.benchmark_gemma import compare_generation_outputs


class BenchmarkHelperTests(unittest.TestCase):
    def test_exact_token_and_text_match_is_completed(self):
        result = compare_generation_outputs(
            {"token_ids": [1, 2, 3], "text": "ok"},
            {"token_ids": [1, 2, 3], "text": "ok"},
        )
        self.assertEqual(
            result,
            {"status": "completed", "token_match": True, "text_match": True},
        )

    def test_token_mismatch_fails_closed(self):
        result = compare_generation_outputs(
            {"token_ids": [1, 2, 3], "text": "ok"},
            {"token_ids": [1, 9, 3], "text": "ok"},
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["token_match"])
        self.assertTrue(result["text_match"])


if __name__ == "__main__":
    unittest.main()
