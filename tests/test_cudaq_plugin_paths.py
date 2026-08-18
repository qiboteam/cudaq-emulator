"""Focused coverage for CUDA-Q plugin paths not covered by qibolab main tests."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import qibolab_cudaq_emulator.controller as controller_module
from qibolab._core.instruments.emulator.engine.abstract import OperatorEvolution
from qibolab._core.pulses import Delay
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper
from qibolab_cudaq_emulator import (
    CudaqEmulatorController,
    CudaqEngine,
    CudaqHamiltonianConfig,
)
from qibolab_cudaq_emulator.hamiltonians import control_operator


NSHOTS = 1000
TARGET_SINGLE_QUBIT_PLATFORM = "qubit-cudaq"


def _controller(platform) -> CudaqEmulatorController:
    controllers = [
        instrument
        for instrument in platform.instruments.values()
        if isinstance(instrument, CudaqEmulatorController)
    ]
    assert len(controllers) == 1
    return controllers[0]


def _require_single_qubit_platform(platform_name: str) -> None:
    if platform_name != TARGET_SINGLE_QUBIT_PLATFORM:
        pytest.skip(
            f"Plugin path assertion is specific to '{TARGET_SINGLE_QUBIT_PLATFORM}', got '{platform_name}'."
        )


def _acquisition_handle(platform, sequence):
    return list(sequence.channel(platform.qubits[0].acquisition))[-1].id


def test_cudaq_platform_executes_with_plugin_controller(platform):
    controller = _controller(platform)
    assert isinstance(controller.engine, CudaqEngine)
    assert isinstance(platform.parameters.configs["hamiltonian"], CudaqHamiltonianConfig)

    sequence = platform.natives.single_qubit[0].MZ()
    acquisition_handle = _acquisition_handle(platform, sequence)
    result = platform.execute([sequence], nshots=NSHOTS)

    assert result[acquisition_handle].shape == (NSHOTS,)


def test_cudaq_batch_sweep_path(platform, platform_name):
    _require_single_qubit_platform(platform_name)
    controller = _controller(platform)
    controller.use_batch_sweep = True

    q0 = platform.natives.single_qubit[0]
    sequence = q0.RX() | q0.MZ()
    drive_channel, drive_pulse = sequence[0]
    acquisition_handle = _acquisition_handle(platform, sequence)
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        values=np.array([0.0, drive_pulse.amplitude], dtype=float),
        pulses=[drive_pulse],
    )

    result = platform.execute([sequence], [[sweeper]], nshots=NSHOTS)

    assert result[acquisition_handle].shape == (NSHOTS, 2)

    frequency_sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([0.0, 1e5], dtype=float),
        channels=[drive_channel],
    )
    frequency_result = platform.execute(
        [sequence], [[frequency_sweeper]], nshots=NSHOTS
    )
    assert frequency_result[acquisition_handle].shape == (NSHOTS, 2)


def test_cudaq_batch_sweep_accepts_single_result(platform, platform_name, monkeypatch):
    _require_single_qubit_platform(platform_name)
    controller = _controller(platform)
    controller.use_batch_sweep = True

    q0 = platform.natives.single_qubit[0]
    sequence = q0.RX() | q0.MZ()
    _, drive_pulse = sequence[0]
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        values=np.array([drive_pulse.amplitude], dtype=float),
        pulses=[drive_pulse],
    )

    monkeypatch.setattr(CudaqEngine, "evolve", lambda self, **_: object())
    monkeypatch.setattr(
        CudaqEngine,
        "get_evolution_states",
        lambda self, _: [None, None],
    )
    monkeypatch.setattr(
        CudaqEngine,
        "get_state_dm",
        lambda self, state, dimensions=None: np.eye(2),
    )

    states, coefficients = controller._sweep(
        sequence.align_to_delays(),
        platform.parameters.configs,
        [[sweeper]],
    )
    assert states.shape == (1, 1, 2, 2)
    assert coefficients is None


def test_cudaq_save_evolution_artifacts(platform, platform_name, tmp_path):
    _require_single_qubit_platform(platform_name)
    controller = _controller(platform)
    controller.save_dir = tmp_path

    q0 = platform.natives.single_qubit[0]
    sequence = q0.RX() | q0.MZ()
    platform.execute([sequence], nshots=NSHOTS)

    assert (tmp_path / "operators.qu").is_file()
    assert (tmp_path / "time_coefficients.npy").is_file()
    assert (tmp_path / "density_matrices.npy").is_file()
    saved_states = sorted(tmp_path.glob("State_Evolution_*.qu"))
    saved_hamiltonians = sorted(tmp_path.glob("System_Hamiltonian_*.qu"))
    assert len(saved_states) == 1
    assert len(saved_hamiltonians) == 1

    with saved_states[0].open("rb") as file_object:
        state_payload = pickle.load(file_object)
    assert len(state_payload.times) == 2
    assert len(state_payload.states) == 2


def test_cudaq_dump_simulation_operator_branches(platform, platform_name, tmp_path, monkeypatch):
    _require_single_qubit_platform(platform_name)
    controller = _controller(platform)
    q0 = platform.natives.single_qubit[0]
    mz_sequence = q0.MZ()
    delay_only_sequence = PulseSequence(
        [(platform.qubits[0].acquisition, Delay(duration=10.0))]
    )

    assert controller._pulse_hamiltonian(
        delay_only_sequence,
        platform.parameters.configs,
    ) is None

    controller.save_dir = tmp_path / "no_pulse"
    controller._dump_simulation(
        mz_sequence,
        platform.parameters.configs,
        np.zeros((1, 2, 2)),
        None,
    )
    assert (controller.save_dir / "operators.qu").is_file()

    monkeypatch.setattr(
        controller_module,
        "hamiltonians",
        lambda *args: [(None, [object()])],
    )
    assert controller._pulse_hamiltonian(
        mz_sequence,
        platform.parameters.configs,
    ) is None

    operator = controller.engine.identity_on_target(0, [2])
    evolution = OperatorEvolution([[operator, lambda time: 1.0]])
    monkeypatch.setattr(
        controller,
        "_pulse_hamiltonian",
        lambda sequence, configs: evolution,
    )
    controller.save_dir = tmp_path / "callable_waveform"
    controller._dump_simulation(
        mz_sequence,
        platform.parameters.configs,
        np.zeros((1, 2, 2)),
        None,
    )
    assert (controller.save_dir / "operators.qu").is_file()


def test_cudaq_flux_control_operator_and_fallback(platform, platform_name):
    _require_single_qubit_platform(platform_name)
    engine = CudaqEngine()
    hamiltonian = platform.parameters.configs["hamiltonian"]
    target = hamiltonian.hilbert_space_index(0)

    flux_operator = control_operator(
        platform.parameters.configs[platform.qubits[0].flux],
        hamiltonian,
        target,
        engine,
    )
    fallback_operator = control_operator(
        platform.parameters.configs[platform.qubits[0].acquisition],
        hamiltonian,
        target,
        engine,
    )

    assert flux_operator is not None
    assert fallback_operator is None

'''
def test_cudaq_coupling_only_hamiltonian_branch(platform, platform_name):
    if platform_name != "split-transmon-coupler-cudaq":
        pytest.skip("Coupling-only branch requires a platform with coupling terms.")

    class CouplingOnlyQubits(dict):
        def items(self):
            return []

    engine = CudaqEngine()
    hamiltonian = platform.parameters.configs["hamiltonian"]
    coupling_only = hamiltonian.model_copy(
        update={"qubits": CouplingOnlyQubits(hamiltonian.qubits)}
    )

    assert coupling_only.hamiltonian(platform.parameters.configs, engine) is not None
'''

def test_cudaq_engine_rungekutta_and_operator_wrappers():
    engine = CudaqEngine(integrator="rungekutta")
    hamiltonian = 0.0 * engine.identity(2)
    initial_state = engine.basis(2, 0)

    assert engine._logical_to_physical_dimensions(None) is None
    assert engine.get_state_dm(initial_state).shape == (2, 2)
    with pytest.raises(ValueError):
        CudaqEngine(integrator="unknown")._integrator([0.0, 0.1])

    time_hamiltonian = OperatorEvolution([[engine.identity(2), lambda time: 1.0]])
    assert engine._compatible_time_hamiltonian(time_hamiltonian) is not None
    assert engine.expand(engine.create(2), targets=0, dims=[2]) is not None
    assert engine.destroy(2) is not None
    assert engine.tensor([engine.identity(2), engine.identity(2)]) is not None

    result = engine.evolve(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        time=[0.0, 0.1],
        collapse_operators=[],
        dimensions={0: 2},
    )
    final_state = engine.get_evolution_states(result)[-1]
    final_amplitudes = np.array(engine.engine.amplitudes(final_state))
    assert int(np.argmax(np.abs(final_amplitudes))) == 0
