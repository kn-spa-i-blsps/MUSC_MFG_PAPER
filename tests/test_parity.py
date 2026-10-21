"""Parity: the ported solver / perception / ladder / params produce BIT-IDENTICAL output to
MUSC_MFG @ physical-refactor-final on the same input. The old side runs from a `git archive` of
the tag (not the working tree), in its own process. Skipped if the old repo is not next to this one
(override with MUSC_MFG_OLD=<path>).
"""
import os
import subprocess
import sys
import tarfile
import io

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_REPO = os.environ.get("MUSC_MFG_OLD", os.path.join(os.path.dirname(ROOT), "MUSC_MFG"))
TAG = "physical-refactor-final"
WEIGHTS_REL = "scratchpad/physical_refactor/phase2b_predictor_weights.npz"


@pytest.fixture(scope="module")
def parity_outputs(tmp_path_factory):
    if not os.path.isdir(os.path.join(OLD_REPO, ".git")):
        pytest.skip(f"old repo not found at {OLD_REPO}")
    tmp = tmp_path_factory.mktemp("parity")
    export = tmp / "old_export"
    export.mkdir()
    blob = subprocess.run(["git", "-C", OLD_REPO, "archive", TAG], check=True,
                          capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(export, filter="data")
    weights = str(export / WEIGHTS_REL)
    script = os.path.join(ROOT, "tests", "parity_compute.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    outs = {}
    for side in ("old", "new"):
        path = str(tmp / f"{side}.npz")
        cmd = [sys.executable, script, "--side", side, "--weights", weights, "--out", path]
        if side == "old":
            cmd += ["--old-root", str(export)]
        subprocess.run(cmd, check=True, env=env, cwd=str(tmp))
        outs[side] = dict(np.load(path))
    return outs


def test_same_keys(parity_outputs):
    assert set(parity_outputs["old"]) == set(parity_outputs["new"])


GROUPS = ("params", "scen", "hjb", "transport", "fp", "cesaro", "perc", "law", "roll")


@pytest.mark.parametrize("group", GROUPS)
def test_bit_identical(parity_outputs, group):
    old, new = parity_outputs["old"], parity_outputs["new"]
    keys = sorted(k for k in old if k.split("/")[0] == group)
    assert keys, f"no outputs in group {group}"
    mismatched = []
    for k in keys:
        a, b = old[k], new[k]
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
            mismatched.append(k)
    assert not mismatched, f"not bit-identical: {mismatched}"
