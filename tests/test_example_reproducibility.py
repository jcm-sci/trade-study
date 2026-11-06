"""Reproducibility checks for published example seed derivation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _seed_from_process(example: str, hash_seed: str) -> str:
    code = (
        "import runpy; "
        f"ns=runpy.run_path({str(ROOT / 'examples' / example)!r}, "
        "run_name='trade_study_seed_test'); "
        "print(ns['stable_config_seed']("
        "{'sample_interval': 15, 'adc_bits': 12, "
        "'sensor_grade': 'lab', 'n_sensors': 2}, stream='test'))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    return subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
    ).strip()


def test_cstr_seed_is_independent_of_python_hash_randomization():
    assert _seed_from_process("cstr_study.py", "1") == _seed_from_process(
        "cstr_study.py", "9341"
    )


def test_monitoring_seed_is_independent_of_python_hash_randomization():
    assert _seed_from_process("monitoring_study.py", "1") == _seed_from_process(
        "monitoring_study.py", "9341"
    )
