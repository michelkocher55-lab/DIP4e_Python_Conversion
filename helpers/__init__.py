from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent

# Preserve legacy `helpers.<module>` imports after splitting helpers into
# library-specific subdirectories.
__path__ = [str(_PACKAGE_DIR)]
for _subdir in ("lib-general", "lib-dip", "lib-dipum"):
    _candidate = _PACKAGE_DIR / _subdir
    if _candidate.is_dir():
        __path__.append(str(_candidate))
