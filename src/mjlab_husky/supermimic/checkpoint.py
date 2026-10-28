"""Buffered, atomic serialization of potentially multi-gigabyte checkpoints."""
import os
from pathlib import Path
import tempfile

import torch


def atomic_torch_save(payload, destination):
    """Commit a complete buffered file, preserving an older file on failure.

    Passing a path directly to PyTorch 2.10 can select its raw FileIO writer
    for non-ASCII paths. A single raw Linux write can be shorter than a large
    storage record. BufferedWriter handles partial raw writes; the temporary
    file is closed and synced before its same-directory atomic replacement.
    The parent directory must already exist. No learner RNG is consumed.
    """
    destination = Path(destination)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                        dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        # Pass the BufferedWriter itself, never its raw handle or file path.
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
