from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _PACKAGE_DIR.parent

# Preserve legacy `lib.<module>` imports used by the converted helper code.
__path__ = [str(_PACKAGE_DIR)]
for _subdir in ("lib-general", "lib-dip", "lib-dipum"):
    _candidate = _ROOT_DIR / "helpers" / _subdir
    if _candidate.is_dir():
        __path__.append(str(_candidate))
