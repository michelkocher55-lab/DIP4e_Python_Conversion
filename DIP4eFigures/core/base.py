from typing import Any
import importlib
from pathlib import Path as _Path
import sys as _sys
import os as _os

import numpy as np
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _MplFigure
from scipy.fftpack import dct


def dip_data(filename: str) -> str:
    """Resolve data files when libDIPUM.data_path is unavailable."""
    env_dir = _os.environ.get("DIP4E_DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(_Path(env_dir).expanduser())
    candidates.append(_Path(__file__).resolve().parents[2] / "AllDataFiles")
    for base in candidates:
        p = base / filename
        if p.exists():
            return str(p)
    searched = "\n".join(f"  - {d}" for d in candidates)
    raise FileNotFoundError(
        f"Could not locate '{filename}'.\n"
        f"Set DIP4E_DATA_DIR to your data directory or place files in one of:\n{searched}"
    )


class DipBase:
    """Shared runtime helpers used by chapter-specific mixins."""

    @staticmethod
    def _patch_ia870_iaintershow() -> None:
        """Patch ia870.iaintershow for NumPy versions without ndarray.tostring."""
        try:
            import ia870 as _ia
        except Exception:
            return

        if getattr(_ia, "_dip4e_iaintershow_patched", False):
            return

        original = getattr(_ia, "iaintershow", None)
        if original is None:
            return

        def _compat_iaintershow(Iab):
            try:
                return original(Iab)
            except AttributeError as exc:
                if "tostring" not in str(exc):
                    raise
                from ia870.iaseunion import iaseunion
                from ia870.iaintersec import iaintersec

                assert (type(Iab) is tuple) and (len(Iab) == 2), (
                    "not proper fortmat of hit-or-miss template"
                )
                A, Bc = Iab
                S = iaseunion(A, Bc)
                Z = iaintersec(S, 0)
                n = np.prod(S.shape)
                one = np.reshape(np.array(n * "1", "c"), S.shape)
                zero = np.reshape(np.array(n * "0", "c"), S.shape)
                x = np.reshape(np.array(n * ".", "c"), S.shape)
                saux = np.choose(S + iaseunion(Z, A), (x, zero, one))
                return "\n".join([ss.tobytes().decode() for ss in saux])

        _ia.iaintershow = _compat_iaintershow
        _ia._dip4e_iaintershow_patched = True

    @staticmethod
    def _patch_ia870_product_compat() -> None:
        """Patch ia870 modules that rely on removed NumPy `product` alias."""
        if not hasattr(np, "product"):
            setattr(np, "product", np.prod)

        module_names = (
            "ia870.iathin",
            "ia870.iathick",
            "ia870.iacthin",
            "ia870.iacthick",
        )
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            if "product" not in module.__dict__:
                module.__dict__["product"] = np.prod

    @staticmethod
    def _patch_ia870_ianeg_overflow_warning() -> None:
        """Patch ia870.ianeg to avoid unsigned-negation overflow warnings."""
        try:
            import ia870 as _ia
            import ia870.ianeg as _ia_ianeg_mod
            from ia870 import ialimits as _ialimits
        except Exception:
            return

        if getattr(_ia, "_dip4e_ianeg_patched", False):
            return

        def _compat_ianeg(f):
            if f.dtype == bool:
                return ~f

            limits = _ialimits(f)
            lo = int(limits[0])
            hi = int(limits[1])
            if lo == -hi:
                y = -f
            else:
                y = lo + hi - f
            return y.astype(f.dtype)

        _ia.ianeg = _compat_ianeg
        _ia_ianeg_mod.ianeg = _compat_ianeg
        _ia._dip4e_ianeg_patched = True

    @staticmethod
    def _patch_numpy_legacy_aliases() -> None:
        """Restore removed NumPy aliases needed by legacy ia870 code."""
        legacy_aliases = {
            "float": float,
            "int": int,
            "complex": complex,
            "object": object,
            "str": str,
        }
        for alias, target in legacy_aliases.items():
            if alias not in np.__dict__:
                setattr(np, alias, target)

    @staticmethod
    def _compute_correlation(g: np.ndarray, w: dict[str, Any]) -> float:
        """Compute correlation between extracted and original watermark coefficients."""
        k = np.size(w["m"])
        g_dct = dct(dct(g.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho")
        coef = np.sort(np.abs(g_dct).ravel())[::-1]
        extracted_coefs = coef[:k]
        original_coefs = np.asarray(w["coef"]).ravel()

        if (
            extracted_coefs.size == 0
            or np.std(extracted_coefs) == 0
            or np.std(original_coefs) == 0
        ):
            return 0.0

        return float(np.corrcoef(extracted_coefs, original_coefs)[0, 1])

    def _prepare_script_context(
        self, data_dir: str | None = None
    ) -> tuple[dict[str, Any], set[int], _Path]:
        """Prepare common execution context for inlined chapter code."""
        self._patch_numpy_legacy_aliases()
        self._patch_ia870_iaintershow()
        self._patch_ia870_product_compat()
        self._patch_ia870_ianeg_overflow_warning()
        project_root = str(_Path(__file__).resolve().parents[2])
        libdipum_dir = str(_Path(__file__).resolve().parents[2] / "libDIPUM")
        had_project_root = project_root in _sys.path
        had_libdipum_dir = libdipum_dir in _sys.path

        old_cwd = _os.getcwd()
        old_data_dir = _os.environ.get("DIP4E_DATA_DIR")
        old_show = _plt.show
        old_pyplot_savefig = _plt.savefig
        old_figure_savefig = _MplFigure.savefig
        old_os_path_exists = _os.path.exists
        pre_fig_nums = set(_plt.get_fignums())
        output_dir = _os.environ.get("DIP4E_OUTPUT_DIR")
        if output_dir is None:
            output_dir = str(_Path(project_root) / "output")

        if data_dir is not None:
            _os.environ["DIP4E_DATA_DIR"] = data_dir
        if not had_project_root:
            _sys.path.insert(0, project_root)
        if not had_libdipum_dir:
            _sys.path.insert(0, libdipum_dir)
        _os.makedirs(output_dir, exist_ok=True)

        def _redirect_save_path(fname: Any) -> Any:
            if isinstance(fname, (str, _os.PathLike)):
                fspath = _os.fspath(fname)
                if not _os.path.isabs(fspath):
                    return _os.path.join(output_dir, fspath)
            return fname

        def _patched_pyplot_savefig(*args, **kwargs):
            if args:
                args = list(args)
                args[0] = _redirect_save_path(args[0])
                args = tuple(args)
            elif "fname" in kwargs:
                kwargs["fname"] = _redirect_save_path(kwargs["fname"])
            return old_pyplot_savefig(*args, **kwargs)

        def _patched_figure_savefig(self, *args, **kwargs):
            if args:
                args = list(args)
                args[0] = _redirect_save_path(args[0])
                args = tuple(args)
            elif "fname" in kwargs:
                kwargs["fname"] = _redirect_save_path(kwargs["fname"])
            return old_figure_savefig(self, *args, **kwargs)

        workspace_mat_names = {
            "Figure112.mat",
            "Figure118.mat",
            "Figure1124.mat",
            "Figure1132.mat",
        }

        def _redirect_workspace_mat_path(path_like: Any) -> Any:
            if not isinstance(path_like, (str, _os.PathLike)):
                return path_like
            p = _os.fspath(path_like)
            if _os.path.isabs(p):
                return p
            if _os.path.dirname(p):
                return p
            if p in workspace_mat_names:
                out_p = _os.path.join(output_dir, p)
                if old_os_path_exists(out_p):
                    return out_p
                all_data_p = _os.path.join(project_root, "AllDataFiles", p)
                if old_os_path_exists(all_data_p):
                    return all_data_p
                return out_p
            return p

        def _patched_exists(path_like: Any) -> bool:
            redirected = _redirect_workspace_mat_path(path_like)
            return old_os_path_exists(redirected)

        import scipy.io as _scipy_io

        old_scipy_loadmat = _scipy_io.loadmat
        old_scipy_savemat = _scipy_io.savemat

        def _patched_loadmat(file_name, *args, **kwargs):
            return old_scipy_loadmat(
                _redirect_workspace_mat_path(file_name), *args, **kwargs
            )

        def _patched_savemat(file_name, *args, **kwargs):
            target = _redirect_workspace_mat_path(file_name)
            if isinstance(target, (str, _os.PathLike)):
                parent = _os.path.dirname(_os.fspath(target))
                if parent:
                    _os.makedirs(parent, exist_ok=True)
            return old_scipy_savemat(target, *args, **kwargs)

        _os.chdir(project_root)
        _plt.show = lambda *args, **kwargs: None
        _plt.savefig = _patched_pyplot_savefig
        _MplFigure.savefig = _patched_figure_savefig
        _os.path.exists = _patched_exists
        _scipy_io.loadmat = _patched_loadmat
        _scipy_io.savemat = _patched_savemat

        state = {
            "old_cwd": old_cwd,
            "old_data_dir": old_data_dir,
            "old_show": old_show,
            "old_pyplot_savefig": old_pyplot_savefig,
            "old_figure_savefig": old_figure_savefig,
            "old_os_path_exists": old_os_path_exists,
            "old_scipy_loadmat": old_scipy_loadmat,
            "old_scipy_savemat": old_scipy_savemat,
            "project_root": project_root,
            "libdipum_dir": libdipum_dir,
            "had_project_root": had_project_root,
            "had_libdipum_dir": had_libdipum_dir,
        }
        script_path = _Path(project_root) / "processing" / "Chapter02" / "_context.py"
        return state, pre_fig_nums, script_path

    def _restore_script_context(
        self, state: dict[str, Any], data_dir: str | None = None
    ) -> None:
        """Restore environment after inlined chapter code execution."""
        _os.chdir(state["old_cwd"])
        _plt.show = state["old_show"]
        _plt.savefig = state["old_pyplot_savefig"]
        _MplFigure.savefig = state["old_figure_savefig"]
        _os.path.exists = state["old_os_path_exists"]
        import scipy.io as _scipy_io

        _scipy_io.loadmat = state["old_scipy_loadmat"]
        _scipy_io.savemat = state["old_scipy_savemat"]

        if data_dir is not None:
            if state["old_data_dir"] is None:
                _os.environ.pop("DIP4E_DATA_DIR", None)
            else:
                _os.environ["DIP4E_DATA_DIR"] = state["old_data_dir"]

        if not state["had_libdipum_dir"] and state["libdipum_dir"] in _sys.path:
            _sys.path.remove(state["libdipum_dir"])
        if not state["had_project_root"] and state["project_root"] in _sys.path:
            _sys.path.remove(state["project_root"])

    def _collect_new_figures(self, pre_fig_nums: set[int]) -> dict[str, Any]:
        """Collect newly created matplotlib figures since `pre_fig_nums`."""
        post_fig_nums = set(_plt.get_fignums())
        new_fig_nums = sorted(post_fig_nums - pre_fig_nums)
        figures = [_plt.figure(num) for num in new_fig_nums]
        return {
            "namespace": {},
            "figures": figures,
            "figure_numbers": new_fig_nums,
        }
