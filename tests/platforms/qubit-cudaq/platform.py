import pathlib

from qibolab import ConfigKinds, DcChannel, IqChannel, Platform, Qubit
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    FluxEmulatorConfig,
    HamiltonianConfig,
)
from qibolab_cudaq_emulator import CudaqEmulatorController


FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])


def create() -> Platform:
    """Create emulator platform with one qubit."""
    qubits = {}
    channels = {}

    for q in range(1):
        qubits[q] = qubit = Qubit.default(q)
        channels |= {
            qubit.drive: IqChannel(mixer=None, lo=None),
            qubit.flux: DcChannel(),
        }

    # register the instruments
    instruments = {
        "dummy": CudaqEmulatorController(
            address="0.0.0.0", channels=channels, sampling_rate_=1
        ),
    }

    return Platform.load(
        path=FOLDER,
        instruments=instruments,
        qubits=qubits,
    )
