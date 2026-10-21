# musc-mfg-paper — Project A

Decentralized MFG routing of a TDMA swarm (macro, soft MFG) with a per-agent hard CBF-QP
safety layer (micro), and delay compensation feeding both layers.

**Provenance:** rebuilt from [kn-spa-i-blsps/MUSC_MFG @ physical-refactor-final](https://github.com/kn-spa-i-blsps/MUSC_MFG/tree/physical-refactor-final)
(commit `c211b1f`). Solver, perception/TDMA model and compensation ladder are ported
faithfully from that tag and checked bit-for-bit by parity tests (`tests/test_parity.py`).

The design is frozen in [`prereg/prereg_projectA.md`](prereg/prereg_projectA.md) — the
single source of truth for parameters, metrics and the smoke gate (§9).

## Layout

```
prereg/        frozen pre-registration
mfg/           HJB, transport, fictitious play, Cesaro FP (ported)
env/           grid + scenarios
control/       params, macro control law, CBF-QP
compensation/  perception / TDMA model, ladder arms (real/cv/learned/smith)
experiments/   config-driven runners (yaml in, results/<run>/ out, config copied alongside)
viz/           animation + diagnostic plots
evidence/      curated artifacts for the paper
results/       run outputs (git-ignored)
tests/         parity + unit tests
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```
