# qibolab-cudaq-emulator

External CUDA-Q emulator plugin for qibolab.

## Compatibility

- qibolab = 0.2.15
- cudaq = 0.14.0
- Python >= 3.10

Compatibility requirements are defined in [pyproject.toml](pyproject.toml).

## Purpose

This package keeps CUDA-Q-specific emulator behavior outside qibolab core while reusing qibolab primitives for:

- pulse sequences
- execution parameters
- emulator result wiring
- waveform and Hamiltonian helpers from qibolab

The plugin provides:

- CudaqEngine
- CudaqHamiltonianConfig
- CudaqEmulatorController

## Installation

From this directory:

```bash
pip install -e .
```

## How to Use in a Platform

Assume you already have a working qibolab emulator platform definition.
Starting from that base setup, switch to this plugin by applying only the changes below.

### 1. Add plugin imports

Keep your existing qibolab emulator imports, then add:

```python
from qibolab_cudaq_emulator import CudaqEmulatorController, CudaqHamiltonianConfig
```

### 2. Register plugin config kind

If your platform uses `ConfigKinds`, extend it to include the plugin Hamiltonian config:

```python
ConfigKinds.extend([CudaqHamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])
```

### 3. Swap emulator controller class

In your `instruments` mapping, replace `EmulatorController(...)` with
`CudaqEmulatorController(...)` and keep your existing channels mapping.

### 4. Swap Hamiltonian config class

Swap the base emulator Hamiltonian config to the CUDA-Q one.

How you do this depends on where your platform stores configs:

- If configs are inline in Python, replace `HamiltonianConfig(...)` with `CudaqHamiltonianConfig(...)`.
- If configs are loaded from platform files (common in qibolab), update your `parameters.json` so:
    - `configs.hamiltonian.kind` is `"cudaq_hamiltonian"`
    - the Hamiltonian payload fields match the CUDA-Q plugin model.

The snippet below shows the key edits in one place:

```python
from qibolab import ConfigKinds
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    FluxEmulatorConfig,
)
from qibolab_cudaq_emulator import CudaqEmulatorController, CudaqHamiltonianConfig

ConfigKinds.extend([CudaqHamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])

instruments = {
    "emulator": CudaqEmulatorController(address="0.0.0.0", channels=channels),
}

Reference platform examples corresponding to those found in qibolab 0.2.15 are available under [tests/platforms](tests/platforms).

## Test Suite

Run tests from this directory:

```bash
pytest -q
```

Notes:

- tests use platform definitions under [tests/platforms](tests/platforms)
- tests skip if cudaq is not available

## Scope

Current focus:

- plugin-local CUDA-Q engine and controller behavior
- plugin-local subsystem remapping and state extraction
- compatibility with qibolab 0.2.15+ platform loading and emulator flow

Out of scope for now:

- preserving every historical branch-only optimization
- documenting benchmark scripts that live outside this public package