"""Buffered checkpoint IO and failure atomicity; large-record check is opt-in."""
import gc
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mjlab_husky.supermimic import checkpoint


class CheckpointTests(unittest.TestCase):
    def test_non_ascii_round_trip_uses_buffered_writer_and_preserves_tensor_views(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint-中文-") as directory:
            destination = Path(directory)/"完整学习状态.pt"
            values = torch.arange(32, dtype=torch.float32)
            payload = {"weights": values, "view": values[4:12], "iteration": 500,
                       "normalizer_count": torch.tensor(123., dtype=torch.float64)}
            original_save = torch.save
            def buffered_save(value, stream):
                self.assertIsInstance(stream, io.BufferedWriter)
                return original_save(value, stream)
            with patch.object(checkpoint.torch, "save", side_effect=buffered_save):
                checkpoint.atomic_torch_save(payload, destination)
            loaded = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(loaded["weights"], values))
            self.assertTrue(torch.equal(loaded["view"], values[4:12]))
            self.assertEqual(loaded["weights"].untyped_storage().data_ptr(),
                             loaded["view"].untyped_storage().data_ptr())
            self.assertEqual(loaded["iteration"], 500)
            self.assertEqual(loaded["normalizer_count"].dtype, torch.float64)
            self.assertEqual(set(Path(directory).iterdir()), {destination})

    def test_serialization_sync_and_replace_failures_preserve_existing_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint-原子-") as directory:
            destination = Path(directory)/"检查点.pt"
            checkpoint.atomic_torch_save({"weights": torch.tensor([7., 9.])}, destination)
            original_bytes = destination.read_bytes()
            def interrupted_save(_payload, stream):
                stream.write(b"incomplete new checkpoint")
                raise OSError("simulated short/disk write failure")
            failures = ((checkpoint.torch, "save", interrupted_save),
                        (checkpoint.os, "fsync", OSError("simulated sync failure")),
                        (checkpoint.os, "replace", OSError("simulated replace failure")))
            for owner, name, failure in failures:
                with self.subTest(stage=name):
                    with patch.object(owner, name, side_effect=failure):
                        with self.assertRaises(OSError):
                            checkpoint.atomic_torch_save({"weights": torch.tensor([100.])}, destination)
                    self.assertEqual(destination.read_bytes(), original_bytes)
                    loaded = torch.load(destination, map_location="cpu", weights_only=False)
                    self.assertTrue(torch.equal(loaded["weights"], torch.tensor([7., 9.])))
                    self.assertEqual(set(Path(directory).iterdir()), {destination})

    def test_failed_first_save_leaves_no_checkpoint_or_partial_file(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint-失败-") as directory:
            destination = Path(directory)/"首次.pt"
            with patch.object(checkpoint.torch, "save", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    checkpoint.atomic_torch_save({"weights": torch.ones(2)}, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(os.environ.get("HUSKY_RUN_LARGE_CHECKPOINT_TEST") == "1",
                         "Opt-in CPU test: needs >2 GiB RAM/disk and writes one >2 GiB storage record")
    def test_non_ascii_single_record_larger_than_two_gib_round_trips(self):
        # One record exceeds Linux's 0x7ffff000 single-write cap. Initialize all
        # bytes so the test never serializes uninitialized memory.
        byte_count = 2**31+4096
        with tempfile.TemporaryDirectory(prefix="checkpoint-大文件-") as directory:
            destination = Path(directory)/"超过两GiB的单条存储.pt"
            values = torch.zeros(byte_count, dtype=torch.uint8, device="cpu")
            sentinels = {0: 17, 0x7ffff000-1: 23, 0x7ffff000: 29, byte_count-1: 31}
            for index, value in sentinels.items():
                values[index] = value
            expected_sha = hashlib.sha256(memoryview(values.numpy())).hexdigest()
            checkpoint.atomic_torch_save({"large_record": values}, destination)
            self.assertGreater(destination.stat().st_size, byte_count)
            del values
            gc.collect()
            loaded = torch.load(destination, map_location="cpu", weights_only=False)["large_record"]
            self.assertEqual(tuple(loaded.shape), (byte_count,))
            self.assertEqual(loaded.dtype, torch.uint8)
            self.assertEqual(loaded.device.type, "cpu")
            for index, value in sentinels.items():
                self.assertEqual(int(loaded[index]), value)
            self.assertEqual(hashlib.sha256(memoryview(loaded.numpy())).hexdigest(), expected_sha)


if __name__ == "__main__":
    unittest.main()
