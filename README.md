# qibolab-cudaq-emulator

External CUDA-Q emulator plugin for qibolab.

## Compatibility

- qibolab = 0.2.15
- cudaq = 0.14.0
- Python >= 3.11

Compatibility requirements are defined in [pyproject.toml](pyproject.toml).

## Purpose

This package keeps CUDA-Q-specific emulator behavior outside qibolab core while reusing qibolab primitives for:

- pulse sequences
- execution parameters
- emulator result wiring
- waveform and Hamiltonian helpers from qibolab

The plugin provides:

- CudaqEngine
- CudaqEmulatorController

## Installation

From this directory:

```bash
pip install -e .
```

If `cudaq==0.14.0` resolves to an unusable CUDA Quantum runtime package on your
system, install the matching runtime wheel explicitly first and then install the
plugin:

```bash
pip install cuda-quantum-cu12==0.14.0
pip install -e .
```

## How to Use in a Platform

Assume you already have a working qibolab emulator platform definition.
Starting from that base setup, switch to this plugin by applying only the changes below.

### 1. Add plugin imports

Keep your existing qibolab emulator imports, then add:

```python
from qibolab_cudaq_emulator import CudaqEmulatorController
```

### 2. Swap emulator controller class

In your `instruments` mapping, replace `EmulatorController(...)` with
`CudaqEmulatorController(...)` and keep your existing channels mapping.

### 3. Keep the existing emulator config kinds

The plugin works with qibolab's standard emulator configs, so you can keep the
existing `HamiltonianConfig`, `DriveEmulatorConfig`, and `FluxEmulatorConfig`
wiring in both Python and `parameters.json`.

The snippet below shows the key edits in one place:

```python
from qibolab import ConfigKinds
from qibolab.instruments.emulator import (
    HamiltonianConfig,
    DriveEmulatorConfig,
    FluxEmulatorConfig,
)
from qibolab_cudaq_emulator import CudaqEmulatorController

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])

instruments = {
    "emulator": CudaqEmulatorController(address="0.0.0.0", channels=channels),
}

Reference platform examples corresponding to those found in qibolab 0.2.15 are available under [tests/platforms](tests/platforms).

## Test Suite

Run tests from this directory:

```bash
pip install -e ".[test]"
pytest -q
```

Notes:

- tests use platform definitions under [tests/platforms](tests/platforms)
- tests skip if cudaq is not available
- tests also require the `test` extra because the mapping diagnostics use `qutip`

## Scope

Current focus:

- plugin-local CUDA-Q engine and controller behavior
- plugin-local subsystem remapping and state extraction
- compatibility with qibolab 0.2.15+ platform loading and emulator flow

Out of scope for now:

- preserving every historical branch-only optimization
- documenting benchmark scripts that live outside this public package
