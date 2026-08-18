from collections import defaultdict
from collections.abc import Iterable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from qibolab._core.components import Config
from qibolab._core.components.configs import AcquisitionConfig
from qibolab._core.instruments.emulator.engine.abstract import (
    OperatorEvolution as AbstractOperatorEvolution,
)
from qibolab._core.instruments.emulator.emulator import (
    EmulatorController,
    SAMPLING_INTERVAL,
    channel_coefficients,
    tlist,
    update_configs,
    update_sequence,
)
from qibolab._core.instruments.emulator.hamiltonians import (
    DriveEmulatorConfig,
    FluxEmulatorConfig,
    HamiltonianConfig,
    Modulated,
    waveform,
)
from qibolab._core.instruments.emulator.results import (
    acquisitions,
    index,
    results as collect_results,
)
from qibolab._core.pulses import Delay, Pulse, PulseLike, VirtualZ
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .engine import CudaqEngine
from .hamiltonians import CudaqHamiltonianConfig, control_operator


def _channel_target_id(channel: str) -> int:
    if channel.startswith("coupler_"):
        return int(channel.split("coupler_", 1)[1].split("/", 1)[0])
    return int(channel.split("/", 1)[0])


class CudaqEmulatorController(EmulatorController):
    """External CUDA-Q emulator controller built on top of qibolab main.

    The plugin overrides only the slices where qibolab main assumes a QuTiP-like
    engine contract.
    """

    engine: CudaqEngine = CudaqEngine()
    initial_state: list[int] | None = None
    use_batch_sweep: bool = False

    def _play_sequence(self, configs, sequence, options, sweepers):
        sweep_results = self._sweep(sequence, configs, sweepers)
        if isinstance(sweep_results, tuple):
            sweep_results = sweep_results[0]

        hamiltonian = cast(HamiltonianConfig, configs["hamiltonian"])
        return collect_results(
            states=sweep_results,
            sequence=sequence,
            hamiltonian=hamiltonian,
            options=options,
        )

    @staticmethod
    def _flatten_batch_specs(specs):
        if isinstance(specs, tuple):
            return [specs]

        flattened = []
        for item in specs:
            flattened.extend(CudaqEmulatorController._flatten_batch_specs(item))
        return flattened

    @staticmethod
    def _measurement_times(sequence: PulseSequence) -> np.ndarray:
        measurement_times = np.array(list(acquisitions(sequence).values()), dtype=float)
        measurement_times[measurement_times < SAMPLING_INTERVAL] = SAMPLING_INTERVAL
        return measurement_times

    @staticmethod
    def _evolution_time_data(
        measurement_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_measurement_times, measurement_indices = np.unique(
            measurement_times, return_inverse=True
        )
        evolution_times = np.concatenate(([0.0], unique_measurement_times))
        return unique_measurement_times, measurement_indices, evolution_times

    def _dump_simulation(self, sequence, configs, states, coefficients):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        config = cast(HamiltonianConfig, configs["hamiltonian"])
        static = config.hamiltonian(config=configs, engine=self.engine)
        evolution = self._pulse_hamiltonian(sequence, configs)
        if evolution is None:
            time_ops = []
        else:
            time_ops = [pair[0] for pair in evolution.operators]

        self.engine.save_operators([static] + time_ops, self.save_dir)
        np.save(self.save_dir / "time_coefficients.npy", coefficients)
        np.save(self.save_dir / "density_matrices.npy", states)

    def _make_sweep_batches(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: dict | None = None,
    ):
        updates = defaultdict(dict) | ({} if updates is None else updates)

        if len(sweepers) == 0:
            sequence_ = update_sequence(sequence, updates)
            configs_ = update_configs(configs, updates)
            config = cast(CudaqHamiltonianConfig, configs_["hamiltonian"])
            hamiltonian = config.hamiltonian(config=configs_, engine=self.engine)
            time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
            time_hamiltonian = self.engine._compatible_time_hamiltonian(time_hamiltonian)
            if time_hamiltonian is not None:
                for operator, waveform_fn in time_hamiltonian.operators:
                    hamiltonian += self.engine.engine.ScalarOperator(waveform_fn) * operator

            measurement_times = self._measurement_times(sequence_)
            _, _, evolution_times = self._evolution_time_data(measurement_times)

            return (hamiltonian, sequence_, configs_, float(evolution_times[-1]))

        parsweep = sweepers[0]
        batch_specs = []
        for values in zip(*(s.values for s in parsweep)):
            sweep_updates = defaultdict(
                dict, {key: value.copy() for key, value in updates.items()}
            )
            for sweeper, value in zip(parsweep, values):
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        sweep_updates[pulse.id].update({sweeper.parameter.name: value})
                if sweeper.channels is not None:
                    for channel in sweeper.channels:
                        sweep_updates[channel].update({sweeper.parameter.name: value})

            batch_specs.append(
                self._make_sweep_batches(sequence, configs, sweepers[1:], sweep_updates)
            )

        return batch_specs

    def _sweep(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        updates: dict | None = None,
    ) -> NDArray | tuple[NDArray, np.ndarray | None]:
        if not self.use_batch_sweep:
            return super()._sweep(sequence, configs, sweepers, updates)

        batch_specs = self._make_sweep_batches(sequence, configs, sweepers, updates)
        config = cast(CudaqHamiltonianConfig, configs["hamiltonian"])
        dimensions = config.hilbert_space_dims()

        flattened_specs = self._flatten_batch_specs(batch_specs)
        hamiltonian_list = [spec[0] for spec in flattened_specs]
        sequence_list = [spec[1] for spec in flattened_specs]
        duration_list = [spec[3] for spec in flattened_specs]

        max_duration_index = int(np.argmax(duration_list))
        max_duration_sequence = sequence_list[max_duration_index]
        max_measurement_times = self._measurement_times(max_duration_sequence)
        _, _, evolution_times = self._evolution_time_data(max_measurement_times)
        saved_times = evolution_times[1:]

        sim_results = self.engine.evolve(
            hamiltonian=hamiltonian_list,
            initial_state=config.initial_state(self.engine),
            time=evolution_times,
            collapse_operators=[config.dissipation(self.engine) for _ in hamiltonian_list],
            time_hamiltonian=None,
            dimensions=dimensions,
        )

        if not isinstance(sim_results, list):
            sim_results = [sim_results]

        sweep_states = []
        for result, batch_sequence in zip(sim_results, sequence_list):
            measurement_times = self._measurement_times(batch_sequence)
            measurement_indices = np.searchsorted(saved_times, measurement_times)
            evolution_states = self.engine.get_evolution_states(result)[1:]

            state_dms = np.stack(
                [
                    self.engine.get_state_dm(state, dimensions=config.dims)
                    for state in evolution_states
                ]
            )
            sweep_states.append(state_dms[measurement_indices])

        sweep_states = np.stack(sweep_states)

        sweepers_shape = [len(sweeper[0].values) for sweeper in sweepers]
        sweepers_shape += list(sweep_states.shape[1:])
        sweep_states = sweep_states.reshape(sweepers_shape)
        return sweep_states, None

    def _evolve(
        self, sequence, configs: dict[str, Config], updates: dict
    ) -> NDArray | tuple[NDArray, np.ndarray | None]:
        sequence_ = update_sequence(sequence, updates)
        configs_ = update_configs(configs, updates)
        config = cast(CudaqHamiltonianConfig, configs_["hamiltonian"])
        hamiltonian = config.hamiltonian(config=configs_, engine=self.engine)
        time_hamiltonian = self._pulse_hamiltonian(sequence_, configs_)
        dimensions = config.hilbert_space_dims()
        measurement_times = self._measurement_times(sequence_)
        _, measurement_indices, evolution_times = self._evolution_time_data(
            measurement_times
        )
        saved_times = evolution_times[1:]
        unique_measurement_indices = np.unique(measurement_indices)
        save_indices = [0, *(index + 1 for index in unique_measurement_indices)]

        sim_results = self.engine.evolve(
            hamiltonian=hamiltonian,
            initial_state=config.initial_state(self.engine),
            time=evolution_times,
            collapse_operators=config.dissipation(self.engine),
            time_hamiltonian=time_hamiltonian,
            dimensions=dimensions,
            save_evolution=self.save_dir,
            save_indices=save_indices,
        )

        evolution_states = self.engine.get_evolution_states(sim_results)[1:]
        states = np.stack(
            [
                self.engine.get_state_dm(state, dimensions=config.dims)
                for state in evolution_states
            ]
        )[measurement_indices]
        coefficients = (
            getattr(time_hamiltonian, "coefficients", None)
            if time_hamiltonian is not None
            else None
        )
        if self.save_dir is not None:
            self._dump_simulation(sequence_, configs_, states, coefficients)
        return states, coefficients

    def _pulse_hamiltonian(self, sequence, configs: dict[str, Config]):
        times = tlist(sequence)
        channels, raw_coefficients = [], []
        for operator, waveforms in hamiltonians(
            sequence, configs, self.engine, self.sampling_rate
        ):
            if operator is None or not waveforms:
                continue
            channels.append(operator)
            raw_coefficients.append(
                channel_coefficients(
                    waveforms,
                    sampling_rate=self.sampling_rate,
                    times=times,
                )
            )

        if not channels:
            return None

        return AbstractOperatorEvolution(
            [
                [operator, coefficient]
                for operator, coefficient in zip(channels, raw_coefficients, strict=True)
            ]
        )
    
def hamiltonian(
    pulses: Iterable[PulseLike],
    config: Config,
    hamiltonian: CudaqHamiltonianConfig,
    hilbert_space_index: int,
    target_id: int,
    engine: CudaqEngine,
    sampling_rate: float,
) -> tuple[object | None, list[Modulated]]:
    op = control_operator(config, hamiltonian, hilbert_space_index, engine)
    waveforms = (
        waveform(pulse, config, hamiltonian.qubits[target_id], sampling_rate)
        for pulse in pulses
        if isinstance(pulse, (Pulse, Delay, VirtualZ))
    )
    return (op, [w for w in waveforms if w is not None])


def hamiltonians(
    sequence: PulseSequence,
    configs: dict[str, Config],
    engine: CudaqEngine,
    sampling_rate: float,
) -> Iterable[tuple[object | None, list[Modulated]]]:
    hconfig = cast(CudaqHamiltonianConfig, configs["hamiltonian"])
    return (
        hamiltonian(
            sequence.channel(channel),
            configs[channel],
            hconfig,
            index(channel, hconfig),
            _channel_target_id(channel),
            engine,
            sampling_rate,
        )
        for channel in sequence.channels
        if not isinstance(configs[channel], AcquisitionConfig)
        if isinstance(configs[channel], (DriveEmulatorConfig, FluxEmulatorConfig))
        if _channel_target_id(channel) in hconfig.qubits
    )
