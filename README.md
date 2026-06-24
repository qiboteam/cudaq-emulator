# qibolab-cudaq-emulator

This directory contains a proposed external CUDA-Q emulator plugin for qibolab `main`.

## Goal

Keep qibolab core unchanged and move CUDA-Q specific behavior into an installable add-on package.

## Design

The plugin reuses qibolab `main` for:

- pulse sequences
- execution parameters
- result processing in `results.py`
- waveform generation in `hamiltonians.py`

The plugin provides its own:

- `CudaqEngine`
- `CudaqHamiltonianConfig`
- `CudaqEmulatorController`

This keeps CUDA-Q specific concerns outside qibolab core:

- logical to physical subsystem remapping
- density matrix extraction from CUDA-Q state payloads
- CUDA-Q scalar callback wrapping for spline waveforms
- target-aware operator construction

## Minimal integration strategy

The proposal deliberately avoids changing qibolab `main` internals. Instead, plugin users instantiate plugin classes directly in their platform definitions.

Example:

```python
from qibolab._core.instruments.emulator.hamiltonians import (
    DriveEmulatorConfig,
    FluxEmulatorConfig,
    Qubit,
    CapacitiveCoupling,
)
from qibolab_cudaq_emulator import CudaqEmulatorController, CudaqHamiltonianConfig

configs = {
    "hamiltonian": CudaqHamiltonianConfig(...),
    "0/drive": DriveEmulatorConfig(...),
    "0/flux": FluxEmulatorConfig(...),
}

instruments = {
    "emulator": CudaqEmulatorController(address="0.0.0.0", channels=channels),
}
```

## Scope of this proposal

This scaffold focuses on the smallest path to an external plugin:

- serial evolution path compatible with qibolab `main`
- plugin-local state extraction and subsystem remapping
- plugin-local Hamiltonian assembly

It does not yet attempt to upstream or preserve every branch-only optimization. In particular, batch sweep execution can be added later as a plugin-only extension after the basic external package shape is accepted.