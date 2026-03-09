from pathlib import Path
import os


ENV_VAR = 'DIP4E_DATA_DIR'
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_data_dirs():
    """Ordered candidate directories that may contain DIP data files."""
    candidates = []

    # 1) User-provided data directory (best for students / different machines).
    env_dir = os.environ.get(ENV_VAR)
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    # 2) Project-local data directory (works in a fresh clone).
    candidates.append(_PROJECT_ROOT / 'AllDataFiles')

    # 3) Legacy absolute dataset directory used on the original machine.
    candidates.append(Path('/Users/michelkocher/michel/Data/DIP-DIPUM/AllDataFiles'))

    return candidates


def dip_data(filename: str) -> str:
    """
    Resolve a DIP data file path in a machine-independent way.

    Usage:
        path = dip_data('mapleleaf.tif')

    Resolution order:
        1) $DIP4E_DATA_DIR
        2) <project_root>/AllDataFiles
        3) /Users/michelkocher/michel/Data/DIP-DIPUM/AllDataFiles
    """
    for base in _candidate_data_dirs():
        p = base / filename
        if p.exists():
            return str(p)

    searched = '\n'.join(f'  - {d}' for d in _candidate_data_dirs())
    raise FileNotFoundError(
        f"Could not locate '{filename}'.\n"
        f"Set {ENV_VAR} to your data directory or place files in one of:\n{searched}"
    )
