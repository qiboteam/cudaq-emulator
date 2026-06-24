from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path


def find_qibolab_test_emulators() -> Path:
    """Locate qibolab main's emulator test platforms without a hardcoded repo path.

    This works best for editable or source-based qibolab installs where the test
    tree is present next to the package sources. Wheel installs may not include
    the upstream test platforms.
    """
    spec = importlib.util.find_spec("qibolab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not import qibolab to locate its source tree.")

    package_path = Path(spec.origin).resolve().parent
    for parent in [package_path, *package_path.parents]:
        candidate = parent / "tests" / "instruments" / "emulator" / "platforms"
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "Could not find qibolab test emulator platforms relative to the installed "
        "qibolab package. If qibolab was installed without tests, provide the "
        "source checkout instead."
    )


def list_platforms(platforms_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in platforms_root.iterdir()
        if path.is_dir() and (path / "platform.py").is_file()
    )


def copy_platforms(
    source_root: Path,
    destination_root: Path,
    names: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    destination_root.mkdir(parents=True, exist_ok=True)

    platform_names = list_platforms(source_root) if names is None else names
    copied = []
    for name in platform_names:
        source = source_root / name
        if not source.is_dir() or not (source / "platform.py").is_file():
            raise FileNotFoundError(f"Platform '{name}' not found in {source_root}.")

        destination = destination_root / name
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination {destination} already exists. Use --overwrite to replace it."
                )
            shutil.rmtree(destination)

        shutil.copytree(source, destination)
        copied.append(destination)

    return copied


def _transform_platform_py(platform_py: Path) -> None:
    text = platform_py.read_text()

    if "from qibolab_cudaq_emulator import" not in text:
        import_block = "from qibolab.instruments.emulator import ("
        start = text.find(import_block)
        if start != -1:
            end = text.find("\n)\n", start)
            if end != -1:
                insert_at = end + len("\n)\n")
                text = (
                    text[:insert_at]
                    + "from qibolab_cudaq_emulator import CudaqHamiltonianConfig, CudaqEmulatorController\n\n"
                    + text[insert_at:]
                )

    # Keep upstream emulator imports intact; only switch the engine/config wiring.
    text = re.sub(
        r"ConfigKinds\.extend\(\[(?P<items>[^\]]+)\]\)",
        lambda match: "ConfigKinds.extend(["
        + match.group("items").replace("HamiltonianConfig", "CudaqHamiltonianConfig")
        + "])",
        text,
    )
    text = text.replace("EmulatorController(", "CudaqEmulatorController(")

    # Drop base emulator imports that become unused after CUDA-Q rewiring.
    text = re.sub(r"^\s*EmulatorController,\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*HamiltonianConfig,\n", "", text, flags=re.MULTILINE)

    platform_py.write_text(text)


def _transform_parameters_json(parameters_json: Path) -> None:
    data = json.loads(parameters_json.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON structure in {parameters_json}.")

    configs = data.get("configs")
    if not isinstance(configs, dict):
        raise ValueError(f"Missing 'configs' object in {parameters_json}.")

    hamiltonian = configs.get("hamiltonian")
    if not isinstance(hamiltonian, dict):
        raise ValueError(f"Missing 'configs.hamiltonian' object in {parameters_json}.")

    hamiltonian["kind"] = "cudaq_hamiltonian"
    parameters_json.write_text(json.dumps(data, indent=2) + "\n")


def sync_cudaq_platforms(
    source_root: Path,
    destination_root: Path,
    names: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    destination_root.mkdir(parents=True, exist_ok=True)

    source_names = list_platforms(source_root) if names is None else names
    synced = []
    for source_name in source_names:
        source = source_root / source_name
        if not source.is_dir() or not (source / "platform.py").is_file():
            raise FileNotFoundError(f"Platform '{source_name}' not found in {source_root}.")

        destination_name = f"{source_name}-cudaq"
        destination = destination_root / destination_name
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination {destination} already exists. Use --overwrite to replace it."
                )
            shutil.rmtree(destination)

        shutil.copytree(source, destination)
        _transform_platform_py(destination / "platform.py")
        _transform_parameters_json(destination / "parameters.json")
        synced.append(destination)

    return synced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate qibolab main's emulator test platforms and optionally copy "
            "them into this plugin repository."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "platforms",
        help="Where to copy the discovered emulator platforms.",
    )
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        help="Platform name to copy. Repeat to copy multiple specific platforms.",
    )
    parser.add_argument(
        "--sync-cudaq-platforms",
        action="store_true",
        help=(
            "Copy all discovered qibolab test emulator platforms into destination "
            "with '-cudaq' suffix and transform them to use the CUDA-Q plugin."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list discovered emulator platforms and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination directories if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = find_qibolab_test_emulators()
    platform_names = list_platforms(source_root)

    print(f"Source: {source_root}")
    if args.list:
        for name in platform_names:
            print(name)
        return 0

    selected = args.platforms if args.platforms is not None else platform_names
    copied = (
        sync_cudaq_platforms(
            source_root=source_root,
            destination_root=args.destination,
            names=selected,
            overwrite=args.overwrite,
        )
        if args.sync_cudaq_platforms
        else copy_platforms(
            source_root=source_root,
            destination_root=args.destination,
            names=selected,
            overwrite=args.overwrite,
        )
    )

    action = "Synced" if args.sync_cudaq_platforms else "Copied"
    print(f"{action} {len(copied)} platform(s) to {args.destination}")
    for path in copied:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())