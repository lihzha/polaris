import base64
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/polaris/verify_pi05_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("verify_pi05_checkpoint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _md5_base64(data: bytes) -> str:
    digest = hashlib.md5(data, usedforsecurity=False).digest()
    return base64.b64encode(digest).decode()


class CheckpointVerificationTest(unittest.TestCase):
    def test_full_then_quick_verification(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "checkpoint"
            root.mkdir()
            payload = b"official checkpoint object"
            (root / "object").write_bytes(payload)
            manifest = Path(temporary_directory) / "manifest.tsv"
            manifest.write_text(
                f"{MODULE.PREFIX}object\t{len(payload)}\t{_md5_base64(payload)}\n"
            )

            full = MODULE.verify_checkpoint(root, manifest, full_md5=True)
            quick = MODULE.verify_checkpoint(root, manifest, full_md5=False)

        self.assertEqual(full["status"], "pass")
        self.assertTrue(full["full_md5"])
        self.assertEqual(quick["object_count"], 1)

    def test_extra_pytorch_checkpoint_path_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "checkpoint"
            root.mkdir()
            payload = b"official checkpoint object"
            (root / "object").write_bytes(payload)
            (root / "model.safetensors").write_bytes(b"wrong model path")
            manifest = Path(temporary_directory) / "manifest.tsv"
            manifest.write_text(
                f"{MODULE.PREFIX}object\t{len(payload)}\t{_md5_base64(payload)}\n"
            )
            with self.assertRaisesRegex(ValueError, "file-set mismatch"):
                MODULE.verify_checkpoint(root, manifest, full_md5=False)

    def test_size_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "checkpoint"
            root.mkdir()
            (root / "object").write_bytes(b"bad")
            manifest = Path(temporary_directory) / "manifest.tsv"
            manifest.write_text(f"{MODULE.PREFIX}object\t4\tunused\n")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                MODULE.verify_checkpoint(root, manifest, full_md5=False)


if __name__ == "__main__":
    unittest.main()
