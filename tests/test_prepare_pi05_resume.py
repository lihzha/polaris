import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/polaris/prepare_pi05_resume.py"
SPEC = importlib.util.spec_from_file_location("prepare_pi05_resume", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareResumePrefixTest(unittest.TestCase):
    def test_completed_prefix_is_copied_and_partial_reset_is_discarded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "eval_results.csv").write_text(
                "episode,episode_length,success,progress,numerical_failure,numerical_failure_reason\n"
                "0,8,False,0.0,False,\n"
                "1,8,True,1.0,False,\n"
            )
            records = [
                {"reset_index": 0, "record_type": "query"},
                {"reset_index": 1, "record_type": "query"},
                {"reset_index": 2, "record_type": "partial"},
            ]
            (source / "policy_traces.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            (source / "episode_0.mp4").write_bytes(b"video-zero")
            (source / "episode_1.mp4").write_bytes(b"video-one")

            summary = MODULE.prepare_resume_prefix(source, destination, 50)

            self.assertEqual(summary["completed_episodes"], 2)
            self.assertEqual(summary["retained_trace_records"], 2)
            self.assertEqual(summary["discarded_partial_trace_records"], 1)
            copied_records = [
                json.loads(line)
                for line in (destination / "policy_traces.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [record["reset_index"] for record in copied_records], [0, 1]
            )
            self.assertEqual(
                (destination / "episode_0.mp4").read_bytes(), b"video-zero"
            )
            self.assertEqual(
                (destination / "episode_1.mp4").read_bytes(), b"video-one"
            )

    def test_extra_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "eval_results.csv").write_text(
                "episode,episode_length\n0,8\n"
            )
            (source / "policy_traces.jsonl").write_text(
                json.dumps({"reset_index": 0}) + "\n"
            )
            (source / "episode_0.mp4").write_bytes(b"complete")
            (source / "episode_1.mp4").write_bytes(b"partial")

            with self.assertRaisesRegex(ValueError, "video set"):
                MODULE.prepare_resume_prefix(source, root / "destination", 50)


if __name__ == "__main__":
    unittest.main()
