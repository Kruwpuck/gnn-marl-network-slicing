"""One wave per tag.

On 2026-08-26 a single launch produced two run_wave processes 9 ms apart, each writing the
same metrics CSVs and checkpoints, with no warning and a zero exit code. The artefacts
survived only because --resume made both writers start from the same checkpoint. These two
assertions are what stops that from being luck next time.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_wave import acquire_wave_lock  # noqa: E402

TAG = "_pytest_lock"


def test_second_wave_with_the_same_tag_refuses_to_start():
    lock = acquire_wave_lock(TAG)
    try:
        assert lock.exists()
        with pytest.raises(SystemExit) as e:
            acquire_wave_lock(TAG)
        assert str(lock) in str(e.value), "the message must name the lock the user has to remove"
        assert "pid" in str(e.value), "and the pid holding it, so removing it is a decision"
    finally:
        lock.unlink(missing_ok=True)


def test_the_lock_is_released_not_permanent():
    """A finished wave must not block the next one -- otherwise the guard becomes the fault."""
    acquire_wave_lock(TAG).unlink()
    lock = acquire_wave_lock(TAG)
    try:
        assert lock.exists()
    finally:
        lock.unlink(missing_ok=True)
