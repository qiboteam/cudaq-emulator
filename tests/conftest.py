from pathlib import Path

import numpy as np
import pytest

import qibolab

PLATFORMS_DIR = Path(__file__).parent / "platforms"
TARGET_PLATFORM_SUFFIX = "-cudaq"


def _cudaq_platform_names() -> list[str]:
    names = sorted(
        path.name
        for path in PLATFORMS_DIR.iterdir()
        if path.is_dir()
        and path.name.endswith(TARGET_PLATFORM_SUFFIX)
        and (path / "platform.py").is_file()
    )
    if not names:
        raise RuntimeError(
            f"No CUDA-Q platform folders matching '*{TARGET_PLATFORM_SUFFIX}' found under {PLATFORMS_DIR}."
        )
    return names


def _single_subsystem_platform_names() -> list[str]:
    return ["qubit-cudaq", "qutrit-cudaq"]


def pytest_generate_tests(metafunc) -> None:
    if "platform_name" in metafunc.fixturenames:
        names = _cudaq_platform_names()
        metafunc.parametrize("platform_name", names, ids=names)


def _create_platform(monkeypatch, platform_name: str) -> qibolab.Platform:
    monkeypatch.setenv("QIBOLAB_PLATFORMS", str(PLATFORMS_DIR))
    return qibolab.create_platform(platform_name)


@pytest.fixture(autouse=True)
def seed() -> None:
    np.random.seed(42)


@pytest.fixture
def platform(monkeypatch, platform_name: str) -> qibolab.Platform:
    return _create_platform(monkeypatch, platform_name)


@pytest.fixture
def split_transmon_coupler_platform(monkeypatch) -> qibolab.Platform:
    return _create_platform(monkeypatch, "split-transmon-coupler-cudaq")


@pytest.fixture(
    params=_single_subsystem_platform_names(),
    ids=_single_subsystem_platform_names(),
)
def single_qubit_platform(monkeypatch, request) -> qibolab.Platform:
    return _create_platform(monkeypatch, request.param)
