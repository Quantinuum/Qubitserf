"""Benchmark protocol tests; run directly without loading a native library."""
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


RUNNER_PATH = Path(__file__).resolve().parents[2] / "bench/distfind/qdistsat_10s.py"
RUNNER = runpy.run_path(str(RUNNER_PATH))
CASE = "BB_108_8_10"


class QDistSATRunnerTests(unittest.TestCase):
    def run_worker(self, script, timeout=0.1, startup_timeout=1):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = RUNNER["run_case"](
                CASE, [sys.executable, "-u", "-c", script], None, output,
                timeout, startup_timeout,
            )
            logs = [(output / result[name]).read_text() for name in ("stdout", "stderr")]
            return result, logs

    def test_pinned_input_hashes(self):
        manifest = RUNNER["verify_inputs"](RUNNER["CASES"])
        self.assertEqual(len(manifest), 26)

    def test_case_filters_preserve_order_and_deduplicate(self):
        selected = RUNNER["select_cases"](["BB_*", CASE])
        self.assertEqual(selected, [case for case in RUNNER["CASES"] if case.startswith("BB_")])
        with self.assertRaises(ValueError):
            RUNNER["select_cases"](["typo"])

    def test_exact_result_validation(self):
        report = {"distance": 10, "proven": True, "backend": "cc", "call_wall_seconds": 0.01}
        for changes, expected in (
            ({}, "completed"),
            ({"distance": 9}, "error"),
            ({"proven": False}, "error"),
            ({"backend": "cuda"}, "error"),
            ({"call_wall_seconds": 0.11}, "timeout"),
            ({"call_wall_seconds": float("nan")}, "error"),
        ):
            with self.subTest(changes=changes):
                status, _ = RUNNER["classify_report"]({**report, **changes}, 10, 0.1)
                self.assertEqual(status, expected)

    def test_partial_startup_line_is_killed_and_logged(self):
        start = time.monotonic()
        record, logs = self.run_worker(
            "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(10)",
            startup_timeout=0.15,
        )
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["phase"], "startup")
        self.assertNotEqual(record["returncode"], 0)
        self.assertEqual(logs[0], "{")

    def test_call_timeout_kills_worker(self):
        record, _ = self.run_worker(
            "import json,sys,time; "
            f"print(json.dumps({{'ready':True,'case':{CASE!r}}})); "
            "sys.stdin.readline(); time.sleep(10)",
        )
        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["phase"], "call")
        self.assertNotEqual(record["returncode"], 0)

    def test_preparation_is_excluded_from_call_budget(self):
        script = (
            "import json,sys,time; time.sleep(0.2); "
            f"print(json.dumps({{'ready':True,'case':{CASE!r}}})); "
            "sys.stdin.readline(); "
            "print(json.dumps({'distance':10,'proven':True,'backend':'cc',"
            "'lower_bound':10,'call_wall_seconds':0.001}))"
        )
        record, _ = self.run_worker(script)
        self.assertEqual(record["status"], "completed")
        self.assertLess(record["observed_wall_seconds"], 0.1)

    def test_worker_error_is_retained(self):
        record, logs = self.run_worker("import sys; print('broken worker',file=sys.stderr); sys.exit(7)")
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["returncode"], 7)
        self.assertIn("broken worker", logs[1])

    def test_worker_rechecks_input_hashes_before_loading(self):
        verify = Mock(side_effect=ValueError("input hash mismatch"))
        worker = RUNNER["worker"]
        with patch.dict(worker.__globals__, verify_inputs=verify):
            with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                worker(SimpleNamespace(worker=CASE))
        verify.assert_called_once_with([CASE])

    def test_malformed_worker_events_are_recorded(self):
        scripts = (
            "print('[]')",
            "import json,sys; "
            f"print(json.dumps({{'ready':True,'case':{CASE!r}}})); "
            "sys.stdin.readline(); print(json.dumps({'call_wall_seconds':None}))",
        )
        for script in scripts:
            with self.subTest(script=script):
                record, _ = self.run_worker(script)
                self.assertEqual(record["status"], "error")
                self.assertIn("error", record)


if __name__ == "__main__":
    unittest.main()
