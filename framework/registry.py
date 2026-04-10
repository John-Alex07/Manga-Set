from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self._items[key] = factory
            return factory

        return decorator

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        if key not in self._items:
            available = ", ".join(sorted(self._items))
            raise KeyError(f"{key!r} is not registered in {self.name}. Available: {available}")
        return self._items[key](*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)


GENERATOR_REGISTRY: Registry[Any] = Registry("generator")
ADAPTER_REGISTRY: Registry[Any] = Registry("adapter")
REFINER_REGISTRY: Registry[Any] = Registry("refiner")
METRIC_REGISTRY: Registry[Any] = Registry("metric")
