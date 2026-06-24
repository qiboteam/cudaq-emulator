import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property, reduce
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qibolab._core.instruments.emulator.engine.abstract import (
    EvolutionResult,
    Operator,
    OperatorEvolution,
    SimulationEngine,
)
from qibolab._core.instruments.emulator.engine.qutip import (
    HAMILTONIAN_FILENAME,
    INTEGRATION_MAX_TIME_STEP,
    INTEGRATION_MIN_TIME_STEP,
    INTEGRATION_MULTIPLIER,
    STATE_FILENAME,
)


@dataclass
class CudaqSavedHamiltonian:
    """Serializable proxy for CUDA-Q Hamiltonians saved with ``save_evolution``."""

    representation: str


@dataclass
class CudaqSavedEvolutionResult:
    """Serializable proxy mirroring the QuTiP result shape used by qibolab."""

    times: list[float]
    states: list[NDArray]
    state_kind: str = "density_matrix"


@dataclass
class CudaqSavedOperators:
    """Serializable proxy for static operator dumps."""

    representations: list[str]


class CudaqEngine(SimulationEngine):
    """CUDA-Q engine kept entirely in the external plugin.

    The plugin does not widen qibolab main's abstract engine contract. Extra
    CUDA-Q-specific helpers stay on this concrete class and are only used by the
    plugin's controller and Hamiltonian helpers.
    """

    max_qubit_dim: int = 3
    base_max_step_size: float = INTEGRATION_MAX_TIME_STEP
    scipyzvode_max_step_size: float = base_max_step_size
    # Empirically calibrated relative to the shared base value: using the same
    # step size as the SciPy ZVODE / QuTiP-aligned path was too coarse on the
    # emulator comparison benchmark, while one fifth of that value provided a
    # better speed-accuracy tradeoff for CUDA-Q Runge-Kutta.
    rungekutta_max_step_size: float = base_max_step_size / 5
    integrator: str = "scipyzvode"

    @cached_property
    def engine(self):
        import cudaq

        cudaq.set_target("dynamics")

        def relax_op_mat(target_dim: int, transition: list[int]):
            def op():
                op_mat = np.zeros([target_dim, target_dim], dtype=np.complex128)
                op_mat[transition[0]][transition[1]] = 1
                return op_mat

            return op

        def deph_op_mat(target_dim: int, pair: list[int]):
            def op():
                op_mat = np.zeros([target_dim, target_dim], dtype=np.complex128)
                op_mat[pair[0]][pair[0]] = 1
                op_mat[pair[1]][pair[1]] = -1
                return op_mat

            return op

        for target_dim in range(2, self.max_qubit_dim + 1):
            for transition in ([nlevel, nlevel + 1] for nlevel in range(target_dim - 1)):
                cudaq.operators.define(
                    f"relax_op_{target_dim}_{transition[0]}_{transition[1]}",
                    [target_dim],
                    relax_op_mat(target_dim, transition),
                    override=True,
                )

            pair = [0, 1]
            cudaq.operators.define(
                f"deph_op_{target_dim}_{pair[0]}_{pair[1]}",
                [target_dim],
                deph_op_mat(target_dim, pair),
                override=True,
            )

        return cudaq

    @staticmethod
    def _logical_to_physical_target(
        target: int, dims: list[int] | None = None
    ) -> int:
        del dims
        return target

    @classmethod
    def _logical_to_physical_dimensions(
        cls, dimensions: dict[int, int] | None = None
    ) -> dict[int, int] | None:
        if dimensions is None:
            return None

        nsubsystems = len(dimensions)
        return {
            physical_index: dimensions[nsubsystems - 1 - physical_index]
            for physical_index in range(nsubsystems)
        }

    @staticmethod
    def _physical_to_logical_density_matrix(
        rho: NDArray, dimensions: list[int] | None = None
    ) -> NDArray:
        if dimensions is None or len(dimensions) <= 1:
            return rho

        nsubsystems = len(dimensions)
        physical_dimensions = list(reversed(dimensions))
        axis_order = list(range(nsubsystems - 1, -1, -1))
        permutation = axis_order + [axis + nsubsystems for axis in axis_order]

        return rho.reshape(*physical_dimensions, *physical_dimensions).transpose(
            permutation
        ).reshape(np.prod(dimensions), np.prod(dimensions))

    @staticmethod
    def _scalar_callback(waveform):
        def callback(t: float) -> float:
            value = waveform(float(np.real(t)))
            return float(np.real(np.asarray(value)).reshape(-1)[0])

        return callback

    @classmethod
    def _compatible_time_hamiltonian(
        cls, time_hamiltonian: OperatorEvolution | None
    ) -> OperatorEvolution | None:
        if time_hamiltonian is None:
            return None

        return OperatorEvolution(
            [
                [operator, cls._scalar_callback(waveform)]
                for operator, waveform in time_hamiltonian.operators
            ]
        )

    @staticmethod
    def _dump_pickle(data: Any, filename: Path) -> None:
        path = filename.with_suffix(filename.suffix + ".qu")
        with path.open("wb") as file_object:
            pickle.dump(data, file_object)

    def save_operators(self, operators, dump_dir: Path) -> None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        self._dump_pickle(
            CudaqSavedOperators(representations=[repr(operator) for operator in operators]),
            dump_dir / "operators",
        )

    def dump_results(
        self,
        hamiltonian: Operator,
        sim_results: EvolutionResult,
        dump_dir: Path,
        time: list[float],
        dimensions: dict[int, int] | None = None,
        save_indices: list[int] | None = None,
    ) -> None:
        """Save CUDA-Q evolution artifacts using the same filenames as QuTiP."""

        dump_dir.mkdir(parents=True, exist_ok=True)

        count_1 = sum(
            1
            for file in dump_dir.iterdir()
            if file.is_file() and HAMILTONIAN_FILENAME in file.name
        )
        count_2 = sum(
            1
            for file in dump_dir.iterdir()
            if file.is_file() and STATE_FILENAME in file.name
        )
        count = max(count_1, count_2)

        statevector_dimension = (
            int(np.prod(list(dimensions.values()))) if dimensions is not None else 0
        )
        evolution_states = self.get_evolution_states(sim_results)
        if save_indices is not None:
            evolution_states = [evolution_states[index] for index in save_indices]
            time = [time[index] for index in save_indices]

        saved_states = [
            self.get_state_dm(state, statevector_dimension=statevector_dimension)
            for state in evolution_states
        ]

        saved_result = CudaqSavedEvolutionResult(
            times=[float(t) for t in time],
            states=saved_states,
        )
        saved_hamiltonian = CudaqSavedHamiltonian(representation=repr(hamiltonian))

        self._dump_pickle(
            saved_hamiltonian,
            dump_dir / f"{HAMILTONIAN_FILENAME}_{count}",
        )
        self._dump_pickle(saved_result, dump_dir / f"{STATE_FILENAME}_{count}")

    def _integrator(self, time: list[float]):
        # ``time`` is only needed for the SciPy ZVODE path so we can derive an
        # ``nsteps`` value consistent with the QuTiP engine default behavior.
        # Runge-Kutta does not expose an independent native ``nsteps`` control;
        # its effective tuning knob here is ``rungekutta_max_step_size``.
        # SciPy ZVODE uses ``scipyzvode_max_step_size`` for the underlying
        # ``max_step`` cap. Both defaults start from the same QuTiP engine
        # constant, but they are configured independently because the two
        # integrators respond very differently to the same nominal step size.
        if self.integrator == "rungekutta":
            return self.engine.RungeKuttaIntegrator(
                max_step_size=self.rungekutta_max_step_size
            )

        if self.integrator == "scipyzvode":
            time_diff = np.diff(time)
            nsteps = max(time_diff) / INTEGRATION_MIN_TIME_STEP * INTEGRATION_MULTIPLIER
            integrator = self.engine.ScipyZvodeIntegrator(nsteps=nsteps)
            integrator.solver._integrator.max_step = self.scipyzvode_max_step_size
            return integrator

        raise ValueError(
            "Unsupported integrator "
            f"{self.integrator!r}. Expected 'scipyzvode' or 'rungekutta'."
        )

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution | None = None,
        collapse_operators: list[Operator] | None = None,
        dimensions: dict[int, int] | None = None,
        save_evolution: Path | None = None,
        save_indices: list[int] | None = None,
        **kwargs,
    ) -> EvolutionResult:
        time = list(time)

        physical_dimensions = self._logical_to_physical_dimensions(dimensions)
        schedule = self.engine.Schedule(time, ["t"])
        time_hamiltonian = self._compatible_time_hamiltonian(time_hamiltonian)

        if time_hamiltonian is not None:
            for operator, waveform in time_hamiltonian.operators:
                hamiltonian += self.engine.ScalarOperator(waveform) * operator

        integrator = self._integrator(time)

        sim_results = self.engine.evolve(
            hamiltonian,
            physical_dimensions,
            schedule,
            initial_state,
            collapse_operators,
            store_intermediate_results=self.engine.IntermediateResultSave.ALL,
            integrator=integrator,
            **kwargs,
        )

        if save_evolution is not None:
            self.dump_results(
                hamiltonian=hamiltonian,
                sim_results=sim_results,
                dump_dir=save_evolution,
                time=time,
                dimensions=dimensions,
                save_indices=save_indices,
            )

        return sim_results

    def create(self, n: int) -> Operator:
        return self.create_on_target(target=0, dims=[n])

    def destroy(self, n: int) -> Operator:
        return self.destroy_on_target(target=0, dims=[n])

    def identity(self, n: int) -> Operator:
        return self.identity_on_target(target=0, dims=[n])

    def create_on_target(self, target: int, dims: list[int] | None = None) -> Operator:
        target = self._logical_to_physical_target(target, dims)
        return self.engine.boson.create(target)

    def destroy_on_target(self, target: int, dims: list[int] | None = None) -> Operator:
        target = self._logical_to_physical_target(target, dims)
        return self.engine.boson.annihilate(target)

    def identity_on_target(
        self, target: int, dims: list[int] | None = None
    ) -> Operator:
        target = self._logical_to_physical_target(target, dims)
        return self.engine.boson.identity(target)

    def tensor(self, operators: list[Operator]) -> Operator:
        return reduce(lambda left, right: left * right, operators)

    def expand(self, op: Operator, targets: int | list[int], dims: list[int]) -> Operator:
        del targets, dims
        return op

    def basis(self, n: int | list[int], state: int | list[int]) -> Operator:
        dims = [n] if isinstance(n, int) else list(n)
        states = [state] if isinstance(state, int) else list(state)

        physical_dims = list(reversed(dims))
        physical_states = list(reversed(states))
        statevec = np.zeros(np.prod(physical_dims))
        statevec[np.ravel_multi_index(physical_states, physical_dims)] = 1.0
        return self.engine.State.from_data(np.array(statevec, dtype=np.complex128))

    def get_state_dm(
        self,
        state: Operator,
        statevector_dimension: int = 0,
        dimensions: list[int] | None = None,
    ) -> NDArray:
        state_data = np.array(self.engine.amplitudes(state), dtype=np.complex128)
        total_dimension = (
            statevector_dimension
            if statevector_dimension
            else int(np.prod(dimensions)) if dimensions is not None else 0
        )

        if state_data.ndim == 2:
            rho = state_data.conj()
        elif total_dimension and len(state_data) == total_dimension**2:
            rho = state_data.reshape([total_dimension, total_dimension]).conj()
        else:
            rho = np.outer(state_data, np.conjugate(state_data))

        rho = self._physical_to_logical_density_matrix(rho, dimensions)
        return 0.5 * (rho + rho.conj().T)

    def get_evolution_states(self, results: EvolutionResult) -> list[Operator]:
        return results.intermediate_states()

    def relaxation_op(self, transition: list[int], target: int, dim: int) -> Operator:
        target = self._logical_to_physical_target(target)
        return self.engine.operators.instantiate(
            f"relax_op_{dim}_{transition[0]}_{transition[1]}", [target]
        )

    def dephasing_op(self, pair: list[int], target: int, dim: int) -> Operator:
        target = self._logical_to_physical_target(target)
        return self.engine.operators.instantiate(
            f"deph_op_{dim}_{pair[0]}_{pair[1]}", [target]
        )