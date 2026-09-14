"""Validated deterministic operation discovery; imported Python is not sandboxed."""

import importlib
import pkgutil
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    name: str
    default: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class Operation:
    id: str
    name: str
    kind: str
    function: object
    input_region: object
    parameters: tuple[Parameter, ...] = ()
    api_version: int = 1
    pixel_formats: tuple[str, ...] = ("RGBA8",)
    locality: str = "regional"
    execution: str = "worker-cooperative"

    def validate_parameters(self, values):
        unknown = set(values) - {parameter.name for parameter in self.parameters}
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        result = {}
        for parameter in self.parameters:
            value = float(values.get(parameter.name, parameter.default))
            if not parameter.minimum <= value <= parameter.maximum:
                raise ValueError(f"{parameter.name} is outside its supported range")
            result[parameter.name] = value
        return result


class Registry:
    def __init__(self):
        self.operations = {}
        self.errors = {}

    def register(self, operation):
        if not isinstance(operation, Operation) or operation.api_version != 1:
            raise ValueError("Unsupported extension API")
        if (
            not operation.id
            or operation.id in self.operations
            or operation.kind not in ("blend", "adjustment", "effect")
            or operation.locality != "regional"
            or operation.execution != "worker-cooperative"
            or not callable(operation.function)
            or not callable(operation.input_region)
            or operation.pixel_formats != ("RGBA8",)
        ):
            raise ValueError(f"Invalid/duplicate operation ID: {operation.id}")
        names = [p.name for p in operation.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate operation parameter")
        operation.validate_parameters({})
        self.operations[operation.id] = operation

    def discover_entry_points(self):
        from importlib.metadata import entry_points

        for entry in sorted(
            entry_points(group="pypaint.operations"), key=lambda item: (item.name, item.value)
        ):
            try:
                self.register(entry.load())
            except Exception as error:
                self.errors[f"entry-point:{entry.name}"] = f"{type(error).__name__}: {error}"

    def discover(self, package):
        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda value: value.name):
            if info.name.startswith("_"):
                continue
            name = package.__name__ + "." + info.name
            try:
                self.register(importlib.import_module(name).OPERATION)
            except Exception as error:
                self.errors[name] = f"{type(error).__name__}: {error}"


def builtins():
    from pypaint.extensions import adjustments, effects

    registry = Registry()
    registry.discover(adjustments)
    registry.discover(effects)
    registry.discover_entry_points()
    return registry
