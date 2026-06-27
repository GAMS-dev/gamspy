from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import gamspy._validation as validation
from gamspy._container import Container
from gamspy._symbols import Equation, Parameter, Set, Variable
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd


class GUSSScenarioDict:
    """
    Helper for constructing a GUSS scenario dictionary set.

    A GUSS scenario dictionary is represented as a normal three-dimensional
    GAMSPy set with rows of the form:

        symbol_name, action, scenario_data_symbol_name
    """

    _ACTIONS: ClassVar[set[str]] = {
        "param",
        "fixed",
        "lower",
        "upper",
        "level",
        "marginal",
        "opt",
    }

    def __init__(
        self,
        container: Container,
        name: str,
        scenario_set: Set,
        description: str = "",
    ):
        if not isinstance(container, Container):
            raise ValidationError("GUSSScenarioDict requires a GAMSPy Container.")

        if not isinstance(name, str):
            raise TypeError("GUSSScenarioDict name must be type str.")

        if name == "":
            raise ValidationError("GUSSScenarioDict name cannot be empty.")

        validation.validate_name(name)

        if not isinstance(scenario_set, Set):
            raise ValidationError("`scenario_set` must be a GAMSPy Set.")

        if scenario_set.container != container:
            raise ValidationError(
                "`scenario_set` must belong to the same container as the "
                "GUSSScenarioDict."
            )

        self.container = container
        self.scenario_set = scenario_set
        self._records: list[tuple[str, str, str]] = [
            (scenario_set.name, "scenario", "")
        ]
        self._entries: set[tuple[str, str]] = set()
        self._set = Set(
            container=container,
            name=name,
            domain=["*", "*", "*"],
            records=self._records,
            description=description,
        )

    @classmethod
    def from_existing(cls, container: Container, name: str) -> GUSSScenarioDict:
        """
        Rebuild a ``GUSSScenarioDict`` wrapper around an already-existing
        underlying ``Set`` in ``container``.

        The Python wrapper is not a container-tracked symbol and is
        not serialized. This classmethod reconstructs the wrapper
        from the surviving Set so that callers can resume using
        ``gp_model.solve(scenario=dict)`` after a deserialize.

        Parameters
        ----------
        container : Container
            The container holding the underlying GUSS scenario Set.
        name : str
            Name of the underlying Set inside ``container``.

        Returns
        -------
        GUSSScenarioDict

        Raises
        ------
        ValidationError
            If ``name`` does not exist in ``container``, is not a ``Set``,
            has no records, or its records do not have the expected
            ``(scenario_set_name, "scenario", "")`` header row.
        """
        if not isinstance(container, Container):
            raise ValidationError("from_existing requires a GAMSPy Container.")
        if not isinstance(name, str) or name == "":
            raise ValidationError("from_existing requires a non-empty `name` string.")

        if name not in container.data:
            raise ValidationError(f"No symbol named `{name}` in the container.")
        underlying = container.data[name]
        if not isinstance(underlying, Set):
            raise ValidationError(f"`{name}` is not a Set ")

        rec = underlying.records
        if rec is None or len(rec) == 0:
            raise ValidationError(f"`{name}` has no records")
        if rec.shape[1] < 3:
            raise ValidationError(
                f"`{name}` has only {rec.shape[1]} column(s); a GUSS "
                f"scenario set requires at least 3 (target, action, "
                f"scenario_data)."
            )

        # Find the scenario-header row by action, not by position
        scenario_row = None
        for row in rec.itertuples(index=False, name=None):
            if str(row[1]) == "scenario":
                scenario_row = row
                break
        if scenario_row is None:
            raise ValidationError(
                f"`{name}` does not look like a GUSS scenario set; no "
                f"record with action='scenario' found."
            )

        header_target = str(scenario_row[0])
        if header_target not in container.data:
            raise ValidationError(
                f"Scenario set `{header_target}` (referenced by `{name}`) "
                f"is not in the container."
            )
        scenario_set_symbol = container.data[header_target]
        if not isinstance(scenario_set_symbol, Set):
            raise ValidationError(
                f"Scenario set `{header_target}` referenced by `{name}` "
                f"is not a Set (got {type(scenario_set_symbol).__name__})."
            )

        instance = cls.__new__(cls)  # bypass __init__; Set already exists
        instance.container = container
        instance.scenario_set = scenario_set_symbol
        instance._set = underlying

        # Rebuild _records and _entries from the Set's records
        rows: list[tuple[str, str, str]] = []
        entries: set[tuple[str, str]] = set()
        for row in rec.itertuples(index=False, name=None):
            target_name = str(row[0])
            action = str(row[1])
            scenario_data_name = str(row[2])
            rows.append((target_name, action, scenario_data_name))
            if action != "scenario":
                entries.add((target_name, action))
        instance._records = rows
        instance._entries = entries

        return instance

    @property
    def name(self) -> str:
        return self._set.name

    @property
    def records(self) -> pd.DataFrame | None:
        return self._set.records

    def to_set(self) -> Set:
        return self._set

    def gamsRepr(self) -> str:
        return self._set.gamsRepr()

    def add_param(self, target: Parameter, scenario_data: Parameter) -> None:
        self._add_entry(target, scenario_data, "param", (Parameter,))

    def add_fixed(self, target: Variable, scenario_data: Parameter) -> None:
        self._add_entry(target, scenario_data, "fixed", (Variable,))

    def add_lower(self, target: Variable, scenario_data: Parameter) -> None:
        self._add_entry(target, scenario_data, "lower", (Variable,))

    def add_upper(self, target: Variable, scenario_data: Parameter) -> None:
        self._add_entry(target, scenario_data, "upper", (Variable,))

    def add_level(self, target: Variable, scenario_data: Parameter) -> None:
        self._add_entry(target, scenario_data, "level", (Variable,))

    def add_marginal(
        self, target: Variable | Equation, scenario_data: Parameter
    ) -> None:
        self._add_entry(target, scenario_data, "marginal", (Variable, Equation))

    def add_options(self, options: Parameter) -> None:
        if not isinstance(options, Parameter):
            raise ValidationError("GUSS options must be a Parameter.")

        self._validate_same_container(options, "options")

        if options.dimension != 1:
            raise ValidationError(
                f"GUSS options parameter `{options.name}` must have dimension 1, "
                f"but found {options.dimension}."
            )

        key = (options.name, "opt")
        if key in self._entries:
            raise ValidationError(
                f"GUSS action `opt` for symbol `{options.name}` already exists."
            )

        self._entries.add(key)
        self._records.append((options.name, "opt", ""))
        self._set.setRecords(self._records)

    def _add_entry(
        self,
        target: Parameter | Variable | Equation,
        scenario_data: Parameter,
        action: str,
        target_types: tuple[type, ...],
    ) -> None:
        if action not in self._ACTIONS:
            raise ValidationError(f"Unsupported GUSS action `{action}`.")

        if not isinstance(target, target_types):
            expected = " or ".join(type_.__name__ for type_ in target_types)
            raise ValidationError(
                f"GUSS action `{action}` requires target type {expected}."
            )

        if not isinstance(scenario_data, Parameter):
            raise ValidationError("GUSS scenario data must be a Parameter.")

        self._validate_same_container(target, "target")
        self._validate_same_container(scenario_data, "scenario_data")
        self._validate_scenario_data_dimension(target, scenario_data, action)

        key = (target.name, action)  # type: ignore[attr-defined]
        if key in self._entries:
            raise ValidationError(
                f"GUSS action `{action}` for symbol `{target.name}` already exists."  # type: ignore[attr-defined]
            )

        self._entries.add(key)
        self._records.append((target.name, action, scenario_data.name))  # type: ignore[attr-defined]
        self._set.setRecords(self._records)

    def _validate_same_container(
        self,
        symbol: Parameter | Variable | Equation,
        role: str,
    ) -> None:
        if symbol.container != self.container:
            raise ValidationError(
                f"GUSS {role} symbol `{symbol.name}` must belong to the same "
                "container as the GUSSScenarioDict."
            )

    def _validate_scenario_data_dimension(
        self,
        target: Parameter | Variable | Equation,
        scenario_data: Parameter,
        action: str,
    ) -> None:
        expected_dimension = self.scenario_set.dimension + target.dimension
        if scenario_data.dimension != expected_dimension:
            raise ValidationError(
                f"GUSS scenario data `{scenario_data.name}` for action `{action}` "
                f"on symbol `{target.name}` must have dimension "
                f"{expected_dimension}, but found {scenario_data.dimension}."
            )
