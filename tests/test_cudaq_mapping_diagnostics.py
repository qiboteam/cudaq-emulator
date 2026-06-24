"""Targeted diagnostics for logical-to-physical subsystem mapping in the CUDA-Q plugin."""

from __future__ import annotations

import numpy as np
import pytest

from qibolab._core.instruments.emulator.engine.qutip import QutipEngine
from qibolab_cudaq_emulator import CudaqEngine

TARGET_PLATFORM = "split-transmon-coupler-cudaq"


def _require_target_platform(platform_name: str) -> None:
    if platform_name != TARGET_PLATFORM:
        pytest.skip(
            f"Mapping diagnostics are specific to '{TARGET_PLATFORM}', got '{platform_name}'."
        )


def test_cudaq_basis_mapping_is_explicit(platform, platform_name):
    _require_target_platform(platform_name)
    engine = CudaqEngine()
    dims = platform.parameters.configs["hamiltonian"].dims

    basis_map = {
        (0, 0, 0): 0,
        (1, 0, 0): 1,
        (0, 1, 0): 2,
        (1, 1, 0): 3,
        (0, 0, 1): 4,
        (1, 0, 1): 5,
        (0, 1, 1): 6,
        (1, 1, 1): 7,
    }

    for logical_state, expected_index in basis_map.items():
        basis_state = engine.basis(dims, list(logical_state))
        amplitudes = np.array(engine.engine.amplitudes(basis_state), dtype=np.complex128)
        assert int(np.argmax(np.abs(amplitudes))) == expected_index


def test_cudaq_pair_01_coupling_matches_reference_subspace(platform, platform_name):
    _require_target_platform(platform_name)
    
    # Extract evolution time from platform's calibrated two-qubit gate duration
    pair = platform.natives.two_qubit[0, 1]
    if pair.iSWAP is None:
        pytest.skip(
            "iSWAP gate not properly calibrated in platform. "
            "Cannot run coupling diagnostics without calibration data."
        )
    sequence = pair.iSWAP()
    if not sequence or len(sequence) == 0:
        pytest.skip(
            "iSWAP gate returned empty sequence. "
            "Cannot run coupling diagnostics without calibration data."
        )
    _, pulse = sequence[0]
    evolution_time = float(pulse.duration)
    
    cudaq_cfg = platform.parameters.configs["hamiltonian"]
    cudaq_engine = CudaqEngine()
    qutip_engine = QutipEngine()

    coupling = cudaq_cfg.pairs[(0, 1)]

    qutip_hamiltonian = qutip_engine.expand(
        coupling.operator(cudaq_cfg.transmon_levels, qutip_engine),
        cudaq_cfg.dims,
        [cudaq_cfg.hilbert_space_index(0), cudaq_cfg.hilbert_space_index(1)],
    )
    qutip_result = qutip_engine.evolve(
        hamiltonian=qutip_hamiltonian,
        initial_state=qutip_engine.basis(cudaq_cfg.dims, [1, 0, 0]),
        time=[0.0, evolution_time],
        collapse_operators=[],
    )
    qutip_state = qutip_result.states[1].full()[:, 0]
    qutip_target_index = int(np.argmax(np.abs(qutip_state)))

    cudaq_hamiltonian = cudaq_cfg._coupling_term((0, 1), coupling, cudaq_engine)
    cudaq_result = cudaq_engine.evolve(
        hamiltonian=cudaq_hamiltonian,
        initial_state=cudaq_engine.basis(cudaq_cfg.dims, [1, 0, 0]),
        time=[0.0, evolution_time],
        collapse_operators=[],
        dimensions={index: dim for index, dim in enumerate(cudaq_cfg.dims)},
    )
    cudaq_state = np.array(
        cudaq_engine.engine.amplitudes(cudaq_engine.get_evolution_states(cudaq_result)[1]),
        dtype=np.complex128,
    )
    cudaq_target_index = int(np.argmax(np.abs(cudaq_state)))

    assert qutip_target_index == 2
    assert cudaq_target_index == qutip_target_index


def test_cudaq_iswap_pulse_transfers_excitation(platform, platform_name):
    _require_target_platform(platform_name)
    q0 = platform.natives.single_qubit[0]
    q1 = platform.natives.single_qubit[1]
    pair = platform.natives.two_qubit[0, 1]

    sequence = q0.RX()
    sequence |= pair.iSWAP()
    sequence |= q0.MZ() + q1.MZ()

    control_handle = list(sequence.channel(platform.qubits[0].acquisition))[-1].id
    target_handle = list(sequence.channel(platform.qubits[1].acquisition))[-1].id
    result = platform.execute([sequence], nshots=2000)

    assert pytest.approx(result[target_handle].mean(), abs=2e-1) == 1
    assert pytest.approx(result[control_handle].mean(), abs=2e-1) == 0
