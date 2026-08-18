from functools import reduce
from operator import add
from typing import Literal

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


class CudaqHamiltonianConfig(HamiltonianConfig):
    """Plugin-local Hamiltonian configuration.

    This class subclasses qibolab main's HamiltonianConfig but overrides the
    engine-facing methods so CUDA-Q-specific operator placement never leaks into
    qibolab core.
    """

    kind: Literal["cudaq_hamiltonian"] = "cudaq_hamiltonian"

    def initial_state(self, engine: CudaqEngine):
        return engine.basis(self.dims, self.nqubits * [0])

    def hilbert_space_dims(self) -> dict[int, int]:
        return {
            self.hilbert_space_index(qubit_id): self.transmon_levels
            for qubit_id in self.qubits
        }

    def _qubit_term(
        self,
        qubit_id: QubitId,
        qubit: Qubit,
        config: dict,
        engine: CudaqEngine,
    ):
        target = self.hilbert_space_index(qubit_id)
        number = engine.create_on_target(target, self.dims) * engine.destroy_on_target(
            target, self.dims
        )
        quartic = (
            engine.create_on_target(target, self.dims)
            * engine.create_on_target(target, self.dims)
            * engine.destroy_on_target(target, self.dims)
            * engine.destroy_on_target(target, self.dims)
        )
        flux = static_flux(qubit=qubit_id, config=config)

        return (
            number * qubit.omega(flux) / 1e9
            + qubit.anharmonicity * 3.141592653589793 * quartic / 1e9
        )

    def _coupling_term(
        self,
        pair_id,
        pair: CapacitiveCoupling,
        engine: CudaqEngine,
    ):
        target1 = self.hilbert_space_index(pair_id[0])
        target2 = self.hilbert_space_index(pair_id[1])
        operator = (
            engine.destroy_on_target(target1, self.dims)
            * engine.create_on_target(target2, self.dims)
            + engine.create_on_target(target1, self.dims)
            * engine.destroy_on_target(target2, self.dims)
        )
        return pair.coupling * 2 * 3.141592653589793 * operator / 1e9

    def hamiltonian(self, config: dict, engine: CudaqEngine):
        qubit_terms = _reduce_operators(
            self._qubit_term(qubit_id, qubit, config, engine)
            for qubit_id, qubit in self.qubits.items()
        )
        coupling_terms = _reduce_operators(
            self._coupling_term(pair_id, pair, engine)
            for pair_id, pair in self.pairs.items()
        )

        if coupling_terms is None:
            return qubit_terms
        return qubit_terms + coupling_terms

    def dissipation(self, engine: CudaqEngine):
        collapse_operators = []
        for qubit_id, qubit in self.qubits.items():
            target = self.hilbert_space_index(qubit_id)
            for transition, t1 in qubit.t1.items():
                collapse_operators.append(
                    (1 / t1) ** 0.5
                    * engine.relaxation_op(
                        transition=list(transition),
                        target=target,
                        dim=self.transmon_levels,
                    )
                )
            for pair, _ in qubit.t2.items():
                collapse_operators.append(
                    (1 / qubit.t_phi(pair) / 2) ** 0.5
                    * engine.dephasing_op(
                        pair=list(pair),
                        target=target,
                        dim=self.transmon_levels,
                    )
                )
        return collapse_operators


__all__ = [
    "CudaqHamiltonianConfig",
    "control_operator",
    "waveform",
]
