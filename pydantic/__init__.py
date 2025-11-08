from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Type, TypeVar

__all__ = ["BaseModel", "Field"]


T = TypeVar("T", bound="BaseModel")


class ValidationError(Exception):
    pass


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<UNSET>"


_UNSET = _UnsetType()


class FieldInfo:
    __slots__ = ("default", "default_factory", "metadata")

    def __init__(
        self,
        default: Any = _UNSET,
        *,
        default_factory: Optional[Callable[[], Any]] = None,
        **metadata: Any,
    ) -> None:
        if default is not _UNSET and default_factory is not None:
            raise ValueError("Field cannot specify both default and default_factory")
        self.default = default
        self.default_factory = default_factory
        self.metadata = metadata

    def has_default(self) -> bool:
        return self.default is not _UNSET or self.default_factory is not None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is not _UNSET:
            return self.default
        return None

    def apply_constraints(self, name: str, value: Any) -> Any:
        if value is None:
            return value
        gt = self.metadata.get("gt")
        ge = self.metadata.get("ge")
        lt = self.metadata.get("lt")
        le = self.metadata.get("le")
        if gt is not None and not value > gt:
            raise ValidationError(f"Field '{name}' must be > {gt}")
        if ge is not None and not value >= ge:
            raise ValidationError(f"Field '{name}' must be >= {ge}")
        if lt is not None and not value < lt:
            raise ValidationError(f"Field '{name}' must be < {lt}")
        if le is not None and not value <= le:
            raise ValidationError(f"Field '{name}' must be <= {le}")
        return value


def Field(
    default: Any = _UNSET,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    **metadata: Any,
) -> FieldInfo:
    """Lightweight replacement for :func:`pydantic.Field`.

    Only the arguments used inside this repository are implemented. Any additional
    keyword arguments are stored as metadata so future consumers can inspect them.
    """

    return FieldInfo(default=default, default_factory=default_factory, **metadata)


class BaseModel:
    __slots__ = ("__dict__",)
    _defaults_cache: ClassVar[Optional[Dict[str, Any]]] = None
    _raw_defaults_cache: ClassVar[Optional[Dict[str, Any]]] = None
    _field_info_cache: ClassVar[Optional[Dict[str, FieldInfo]]] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._defaults_cache = None
        cls._raw_defaults_cache = None
        cls._field_info_cache = None

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        values: Dict[str, Any] = {}
        defaults = self._collect_defaults()
        field_info = self._collect_field_info()
        for name, _ in annotations.items():
            if name in data:
                values[name] = data[name]
            elif name in field_info and field_info[name].has_default():
                values[name] = field_info[name].get_default()
            elif name in defaults:
                values[name] = defaults[name]
            else:
                values[name] = None
            if name in field_info:
                values[name] = field_info[name].apply_constraints(name, values[name])
        for key, value in data.items():
            if key not in values:
                values[key] = value
        self.__dict__ = values

    @classmethod
    def _collect_defaults(cls) -> Dict[str, Any]:
        cache = getattr(cls, "_defaults_cache", None)
        if cache is not None:
            return cache
        raw_defaults = cls._collect_raw_defaults()
        field_info = cls._collect_field_info()
        defaults: Dict[str, Any] = {}
        for name in cls._collect_annotations():
            if name in field_info and field_info[name].has_default():
                defaults[name] = field_info[name].get_default()
            elif name in raw_defaults:
                defaults[name] = raw_defaults[name]
        cls._defaults_cache = defaults
        return defaults

    @classmethod
    def _collect_raw_defaults(cls) -> Dict[str, Any]:
        cache = getattr(cls, "_raw_defaults_cache", None)
        if cache is not None:
            return cache
        defaults: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            defaults.update({k: v for k, v in getattr(base, "__dict__", {}).items() if not k.startswith("__")})
        for name, value in list(defaults.items()):
            if isinstance(value, FieldInfo):
                defaults.pop(name, None)
        annotations = getattr(cls, "__annotations__", {})
        filtered = {k: defaults.get(k) for k in annotations.keys() if k in defaults}
        cls._raw_defaults_cache = filtered
        return filtered

    @classmethod
    def _collect_field_info(cls) -> Dict[str, FieldInfo]:
        cache = getattr(cls, "_field_info_cache", None)
        if cache is not None:
            return cache
        info: Dict[str, FieldInfo] = {}
        for base in reversed(cls.__mro__):
            namespace = getattr(base, "__dict__", {})
            for name, value in namespace.items():
                if isinstance(value, FieldInfo):
                    info[name] = value
        annotations = getattr(cls, "__annotations__", {})
        filtered = {k: info[k] for k in annotations.keys() if k in info}
        cls._field_info_cache = filtered
        return filtered

    @classmethod
    def _collect_annotations(cls) -> Dict[str, Any]:
        annotations: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            annotations.update(getattr(base, "__annotations__", {}))
        return annotations

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

