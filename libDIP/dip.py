from typing import Any
from pathlib import Path
from pathlib import Path as _Path
import sys
import os
import os as _os

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIP.core import DipBase, dip_data
from libDIP.chapters import Chapter02Mixin, Chapter03Mixin, Chapter04Mixin, Chapter05Mixin, Chapter06Mixin, Chapter07Mixin, Chapter08Mixin, Chapter09Mixin, Chapter10Mixin, Chapter11Mixin, Chapter12Mixin, Chapter13Mixin
from libDIP.watermark4e import watermark4e


class _BoundChapter:
    def __init__(self, owner: "Dip", mixin_cls: type) -> None:
        self._owner = owner
        self._mixin_cls = mixin_cls

    def __getattr__(self, name: str):
        if not hasattr(self._mixin_cls, name):
            raise AttributeError(
                f"{self._mixin_cls.__name__!s} has no attribute {name!r}"
            )
        return getattr(self._owner, name)


class Dip(Chapter02Mixin, Chapter03Mixin, Chapter04Mixin, Chapter05Mixin, Chapter06Mixin, Chapter07Mixin, Chapter08Mixin, Chapter09Mixin, Chapter10Mixin, Chapter11Mixin, Chapter12Mixin, Chapter13Mixin, DipBase):
    """High-level facade composed from chapter-specific mixins."""

    @property
    def chapter02(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter02Mixin)

    @property
    def chapter03(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter03Mixin)

    @property
    def chapter04(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter04Mixin)

    @property
    def chapter05(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter05Mixin)

    @property
    def chapter06(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter06Mixin)

    @property
    def chapter07(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter07Mixin)

    @property
    def chapter08(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter08Mixin)

    @property
    def chapter09(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter09Mixin)

    @property
    def chapter10(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter10Mixin)

    @property
    def chapter11(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter11Mixin)

    @property
    def chapter12(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter12Mixin)

    @property
    def chapter13(self) -> _BoundChapter:
        return _BoundChapter(self, Chapter13Mixin)

    def figure850(
        self,
        watermark_size: int = 1000,
        seed: int = 123,
        image_name: str = "lena.tif",
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        """Run Figure 8.50 watermark processing and return result arrays."""
        np.random.seed(seed)
        old_data_dir = os.environ.get("DIP4E_DATA_DIR")
        if data_dir is not None:
            os.environ["DIP4E_DATA_DIR"] = data_dir
        try:
            f = imread(dip_data(image_name))
        finally:
            if data_dir is not None:
                if old_data_dir is None:
                    os.environ.pop("DIP4E_DATA_DIR", None)
                else:
                    os.environ["DIP4E_DATA_DIR"] = old_data_dir
        if f.ndim == 3:
            f = f[..., 0]
        f = f.astype(np.uint8)

        m1 = np.random.randn(watermark_size)
        m2 = np.random.randn(watermark_size)

        g1, w1 = watermark4e(f, m1)
        diff1 = g1.astype(float) - f.astype(float)

        g2, w2 = watermark4e(f, m2)
        diff2 = g2.astype(float) - f.astype(float)

        r1 = self._compute_correlation(g1, w1)
        r2 = self._compute_correlation(g2, w2)

        return {
            "image": f,
            "g1": g1,
            "g2": g2,
            "diff1": diff1,
            "diff2": diff2,
            "r1": r1,
            "r2": r2,
            "w1": w1,
            "w2": w2,
        }

