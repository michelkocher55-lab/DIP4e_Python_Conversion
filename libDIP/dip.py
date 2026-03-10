from libDIP.core import DipBase
from libDIP.chapters import Chapter02Mixin, Chapter03Mixin, Chapter04Mixin, Chapter05Mixin, Chapter06Mixin, Chapter07Mixin, Chapter08Mixin, Chapter09Mixin, Chapter10Mixin, Chapter11Mixin, Chapter12Mixin, Chapter13Mixin


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
