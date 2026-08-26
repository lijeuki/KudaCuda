import unittest
from types import SimpleNamespace
from unittest.mock import patch

from KudaCuda.tools.benchmark.audit_runtime import audit_runtime, capture_gpu_state, warm_up_cuda


class FakeQuantizedModule:
    def __init__(self):
        self.weight = SimpleNamespace(
            quant_state=SimpleNamespace(
                quant_type="nf4",
                blocksize=64,
                nested=False,
            )
        )


class FakeModel:
    def modules(self):
        return [self, FakeQuantizedModule()]


class RuntimeAuditTests(unittest.TestCase):
    def test_cuda_warmup_is_skipped_for_non_cuda_devices(self):
        result = warm_up_cuda(SimpleNamespace(), SimpleNamespace(type="cpu"), 10.0)

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "non_cuda_device")

    def test_audit_reports_requested_nf4_dtype_and_registered_cuda_ops(self):
        fake_torch = SimpleNamespace(
            __version__="2.11.0",
            version=SimpleNamespace(cuda="12.8"),
            ops=SimpleNamespace(
                bitsandbytes=SimpleNamespace(
                    gemv_4bit=object(),
                    gemm_4bit=object(),
                )
            ),
        )
        args = SimpleNamespace(dtype="float16", load_in_4bit=True)

        result = audit_runtime(fake_torch, FakeModel(), args)

        self.assertEqual(result["requested_compute_dtype"], "float16")
        self.assertTrue(result["quantization_requested"])
        self.assertEqual(result["quantized_module_count"], 1)
        self.assertEqual(result["registered_bitsandbytes_ops"], ["gemm_4bit", "gemv_4bit"])

    @patch("KudaCuda.tools.benchmark.audit_runtime.subprocess.run")
    def test_capture_gpu_state_parses_nvidia_smi_csv(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="RTX 3050, P0, 1500 MHz, 6001 MHz, 47, 11.50 W\n",
        )

        result = capture_gpu_state()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["raw"], "RTX 3050, P0, 1500 MHz, 6001 MHz, 47, 11.50 W")
        self.assertEqual(result["fields"]["pstate"], "P0")


if __name__ == "__main__":
    unittest.main()
