from functools import reduce
from operator import add

from qibolab._core.components import Config
from qibolab._core.identifier import QubitId
from qibolab._core.instruments.emulator.hamiltonians import (
    CapacitiveCoupling,
    DriveEmulatorConfig,
    FluxEmulatorConfig,
    HamiltonianConfig,
    Qubit,
    static_flux,
    waveform,
)

from .engine import CudaqEngine


def _reduce_operators(operators):
    operators = list(operators)
    if not operators:
        return None
    return reduce(add, operators[1:], operators[0])


def control_operator(
    config: Config,
    hamiltonian: HamiltonianConfig,
    target: int,
    engine: CudaqEngine,
):
    if isinstance(config, DriveEmulatorConfig):
        return -1j * (
            engine.destroy_on_target(target, hamiltonian.dims)
            - engine.create_on_target(target, hamiltonian.dims)
        )

    if isinstance(config, FluxEmulatorConfig):
        return engine.create_on_target(target, hamiltonian.dims) * engine.destroy_on_target(
            target, hamiltonian.dims
        )

    return None


def initial_state(hamiltonian: HamiltonianConfig, engine: CudaqEngine):
    return engine.basis(hamiltonian.dims, hamiltonian.nqubits * [0])


def hilbert_space_dims(hamiltonian: HamiltonianConfig) -> dict[int, int]:
    return {
        hamiltonian.hilbert_space_index(qubit_id): hamiltonian.transmon_levels
        for qubit_id in hamiltonian.qubits
    }


def qubit_term(
    hamiltonian: HamiltonianConfig,
    qubit_id: QubitId,
    qubit: Qubit,
    config: dict,
    engine: CudaqEngine,
):
    target = hamiltonian.hilbert_space_index(qubit_id)
    number = engine.create_on_target(target, hamiltonian.dims) * engine.destroy_on_target(
        target, hamiltonian.dims
    )
    quartic = (
        engine.create_on_target(target, hamiltonian.dims)
        * engine.create_on_target(target, hamiltonian.dims)
        * engine.destroy_on_target(target, hamiltonian.dims)
        * engine.destroy_on_target(target, hamiltonian.dims)
    )
    flux = static_flux(qubit=qubit_id, config=config)

    return (
        number * qubit.omega(flux) / 1e9
        + qubit.anharmonicity * 3.141592653589793 * quartic / 1e9
    )


def coupling_term(
    hamiltonian: HamiltonianConfig,
    pair_id,
    pair: CapacitiveCoupling,
    engine: CudaqEngine,
):
    target1 = hamiltonian.hilbert_space_index(pair_id[0])
    target2 = hamiltonian.hilbert_space_index(pair_id[1])
    operator = (
        engine.destroy_on_target(target1, hamiltonian.dims)
        * engine.create_on_target(target2, hamiltonian.dims)
        + engine.create_on_target(target1, hamiltonian.dims)
        * engine.destroy_on_target(target2, hamiltonian.dims)
    )
    return pair.coupling * 2 * 3.141592653589793 * operator / 1e9


def static_hamiltonian(
    hamiltonian: HamiltonianConfig,
    config: dict,
    engine: CudaqEngine,
):
    qubit_terms = _reduce_operators(
        qubit_term(hamiltonian, qubit_id, qubit, config, engine)
        for qubit_id, qubit in hamiltonian.qubits.items()
    )
    coupling_terms = _reduce_operators(
        coupling_term(hamiltonian, pair_id, pair, engine)
        for pair_id, pair in hamiltonian.pairs.items()
    )

    if coupling_terms is None:
        return qubit_terms
    return qubit_terms + coupling_terms


def dissipation(hamiltonian: HamiltonianConfig, engine: CudaqEngine):
    collapse_operators = []
    for qubit_id, qubit in hamiltonian.qubits.items():
        target = hamiltonian.hilbert_space_index(qubit_id)
        for transition, t1 in qubit.t1.items():
            collapse_operators.append(
                (1 / t1) ** 0.5
                * engine.relaxation_op(
                    transition=list(transition),
                    target=target,
                    dim=hamiltonian.transmon_levels,
                )
            )
        for pair, _ in qubit.t2.items():
            collapse_operators.append(
                (1 / qubit.t_phi(pair) / 2) ** 0.5
                * engine.dephasing_op(
                    pair=list(pair),
                    target=target,
                    dim=hamiltonian.transmon_levels,
                )
            )
    return collapse_operators


__all__ = [
    "coupling_term",
    "control_operator",
    "dissipation",
    "hilbert_space_dims",
    "initial_state",
    "qubit_term",
    "static_hamiltonian",
    "waveform",
]
