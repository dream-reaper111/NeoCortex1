from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Optional, Type, TypeVar

__all__ = ["BaseModel"]


T = TypeVar("T", bound="BaseModel")


class ValidationError(Exception):
    pass


class BaseModel:
    __slots__ = ("__dict__",)
    _defaults_cache: ClassVar[Optional[Dict[str, Any]]] = None

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        values: Dict[str, Any] = {}
        defaults = self._collect_defaults()
        for name, _ in annotations.items():
            if name in data:
                values[name] = data[name]
            elif name in defaults:
                values[name] = defaults[name]
            else:
                values[name] = None
        for key, value in data.items():
            if key not in values:
                values[key] = value
        self.__dict__ = values

    @classmethod
    def _collect_defaults(cls) -> Dict[str, Any]:
        cache = getattr(cls, "_defaults_cache", None)
        if cache is not None:
            return cache
        defaults: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            defaults.update({k: v for k, v in getattr(base, "__dict__", {}).items() if not k.startswith("__")})
        annotations = getattr(cls, "__annotations__", {})
        filtered = {k: defaults.get(k) for k in annotations.keys() if k in defaults}
        cls._defaults_cache = filtered
        return filtered

    @classmethod
    def parse_obj(cls: Type[T], obj: Mapping[str, Any]) -> T:
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, Mapping):
            raise ValidationError("parse_obj requires a mapping")
        return cls(**dict(obj))

    def dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def model_dump(self) -> Dict[str, Any]:
        return self.dict()

