from __future__ import annotations

import unittest

from KudaCuda.tools.chat_qwen2_kudacuda import (
    DEFAULT_TACTIC,
    activation_snapshot,
    build_messages,
    native_kv_preflight,
    normalize_model_reference,
    _parse_args,
    resolve_sidecar_python_dir,
    summarize_turn,
)


class FakeBridge:
    native_attention_dispatcher = True
    native_attention_dispatches = 12
    last_qwen2_dispatch_reason = "native"

    def tactic_status(self, tactic_name: str) -> str:
        self.requested_tactic = tactic_name
        return "implemented"


class ChatToolTests(unittest.TestCase):
    def test_streaming_is_enabled_by_default_and_can_be_disabled(self) -> None:
        self.assertTrue(_parse_args(["--model", "Qwen/test"]).stream)
        self.assertFalse(_parse_args(["--model", "Qwen/test", "--no-stream"]).stream)

    def test_normalize_model_reference_removes_accidental_terminal_line_wrap(self) -> None:
        wrapped = (
            "D:\\AI\\Models\\models--Qwen--Qwen2.5-3B-\n"
            "    Instruct\\snapshots\\revision"
        )

        self.assertEqual(
            normalize_model_reference(wrapped),
            "D:\\AI\\Models\\models--Qwen--Qwen2.5-3B-Instruct\\snapshots\\revision",
        )

    def test_resolve_sidecar_python_dir_falls_back_to_script_repository(self) -> None:
        sidecar_dir = resolve_sidecar_python_dir("C:/missing-kudacuda-root")

        self.assertTrue((sidecar_dir / "kudacuda_sidecar.py").is_file())

    def test_build_messages_preserves_history_and_adds_custom_user_message(self) -> None:
        history = [
            {"role": "user", "content": "Remember that I prefer concise answers."},
            {"role": "assistant", "content": "Understood."},
        ]

        messages = build_messages(
            history,
            "Explain CUDA Graphs.",
            system_prompt="You are a CUDA assistant.",
        )

        self.assertEqual(messages[0], {"role": "system", "content": "You are a CUDA assistant."})
        self.assertEqual(messages[1:], history + [{"role": "user", "content": "Explain CUDA Graphs."}])

    def test_activation_snapshot_reports_native_dispatch_capability(self) -> None:
        bridge = FakeBridge()

        snapshot = activation_snapshot(bridge, DEFAULT_TACTIC)

        self.assertTrue(snapshot["native_attention_dispatcher"])
        self.assertEqual(snapshot["tactic_status"], "implemented")
        self.assertEqual(snapshot["native_attention_dispatches"], 12)
        self.assertEqual(snapshot["last_dispatch_reason"], "native")

    def test_summarize_turn_reports_dispatch_delta_and_throughput(self) -> None:
        bridge = FakeBridge()

        summary = summarize_turn(
            bridge,
            before_dispatches=5,
            generated_tokens=10,
            elapsed_seconds=2.0,
        )

        self.assertEqual(summary["native_dispatches"], 7)
        self.assertEqual(summary["generated_tokens"], 10)
        self.assertEqual(summary["tokens_per_second"], 5.0)
        self.assertEqual(summary["last_dispatch_reason"], "native")

    def test_native_kv_preflight_reports_when_a_turn_will_leave_the_native_profile(self) -> None:
        preflight = native_kv_preflight(
            prompt_tokens=500,
            requested_tokens=128,
            native_limit=512,
        )

        self.assertEqual(preflight["native_budget"], 12)
        self.assertTrue(preflight["will_fallback"])
        self.assertEqual(preflight["fallback_after_tokens"], 12)

    def test_native_kv_preflight_marks_extended_profile_as_native(self) -> None:
        preflight = native_kv_preflight(
            prompt_tokens=585,
            requested_tokens=128,
            native_limit=2048,
        )

        self.assertEqual(preflight["native_budget"], 1463)
        self.assertFalse(preflight["will_fallback"])
        self.assertIsNone(preflight["fallback_after_tokens"])


if __name__ == "__main__":
    unittest.main()
