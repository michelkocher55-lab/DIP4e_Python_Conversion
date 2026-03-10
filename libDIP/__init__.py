from pathlib import Path as _Path

# Compatibility path for modules moved out of `libDIP/`.
_PKG_DIR = _Path(__file__).resolve().parent
for _extra_dir in (
    _PKG_DIR.parent / "helpers",
    _PKG_DIR.parent / "helpers" / "lib-general",
    _PKG_DIR.parent / "helpers" / "lib-dip",
    _PKG_DIR.parent / "helpers" / "lib-dipum",
    _PKG_DIR.parent / "libDIP-full",
):
    if _extra_dir.is_dir():
        _extra_dir_str = str(_extra_dir)
        if _extra_dir_str not in __path__:
            __path__.append(_extra_dir_str)

from .dip import Dip

__all__ = ["Dip"]
