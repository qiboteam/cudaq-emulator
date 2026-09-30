import pathlib

from qibolab import (
    AcquisitionChannel,
    ConfigKinds,
    DcChannel,
    IqChannel,
    Platform,
    Qubit,
    QubitMap,
)
from qibolab.instruments.emulator import (
    DriveEmulatorConfig,
    FluxEmulatorConfig,
    HamiltonianConfig,
)
from qibolab_cudaq_emulator import CudaqEmulatorController


FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig, FluxEmulatorConfig])


def create() -> Platform:
    """Create emulator platform with one qutrit."""
    qubits: QubitMap = {}
    channels = {}

    for q in range(1):
        qubits[q] = qubit = Qubit.default(q, drive_extra={(1, 2): f"{q}/drive12"})
        channels |= {
            qubit.acquisition: AcquisitionChannel(probe=qubit.probe),
            qubit.drive: IqChannel(mixer=None, lo=None),
            qubit.drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
            qubit.flux: DcChannel(),
        }

    # register the instruments
    instruments = {
        "dummy": CudaqEmulatorController(address="0.0.0.0", channels=channels),
    }

    return Platform.load(
        path=FOLDER,
        instruments=instruments,
        qubits=qubits,
    )
