from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import gamspy as gp
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Set, Parameter, Variable, Equation


class MiroJSONEncoder:
    def __init__(
        self,
        container: Container,
        input_symbols: list[str],
        output_symbols: list[str],
    ):
        self.container = container
        self.model_title = "GAMSPy App"
        self.input_symbols = input_symbols
        self.output_symbols = output_symbols
        self.input_scalars = self._find_scalars(input_symbols)
        self.output_scalars = self._find_scalars(output_symbols)
        self.miro_json = self.prepare_json()

    def _find_scalars(self, symbols: list[str]) -> list[str]:
        scalars = []
        for name in symbols:
            symbol = self.container[name]

            if len(symbol.domain) == 0:
                scalars.append(name)

            if isinstance(symbol, gp.Set) and symbol.is_singleton:
                scalars.append(name)

        return scalars

    def _prepare_scalars(
        self, alias: str, symbols: list[str]
    ) -> tuple[dict | None, dict | None]:
        sp_names, sp_texts, sp_types = [], [], []
        ve_names, ve_texts, ve_types = [], [], []

        for name in symbols:
            symbol = self.container[name]

            if isinstance(symbol, (gp.Set, gp.Parameter)):
                sp_names.append(name)
                sp_texts.append(
                    symbol.description if symbol.description else symbol.name
                )
                sp_types.append(type(symbol).__name__.lower())
            elif isinstance(symbol, (gp.Variable, gp.Equation)):
                ve_names.append(name)
                ve_texts.append(
                    symbol.description if symbol.description else symbol.name
                )
                ve_types.append(type(symbol).__name__.lower())

        sp_scalars_dict = None
        ve_scalars_dict = None

        if len(sp_names) != 0:
            sp_scalars_dict = {
                "alias": alias,
                "symnames": sp_names,
                "symtext": sp_texts,
                "symtypes": sp_types,
                "headers": {
                    "scalar": {
                        "type": "string",
                        "alias": "Scalar Name",
                    },
                    "description": {
                        "type": "string",
                        "alias": "Scalar Description",
                    },
                    "value": {
                        "type": "string",
                        "alias": "Scalar Value",
                    },
                },
            }

        if len(ve_names) != 0:
            ve_scalars_dict = {
                "alias": "Output Variable/Equation Scalars",
                "symnames": ve_names,
                "symtext": ve_texts,
                "symtypes": ve_types,
                "headers": {
                    "scalar": {"type": "string", "alias": "Scalar Name"},
                    "description": {
                        "type": "string",
                        "alias": "Scalar Description",
                    },
                    "level": {"type": "numeric", "alias": "Level"},
                    "marginal": {"type": "numeric", "alias": "Marginal"},
                    "lower": {"type": "numeric", "alias": "Lower"},
                    "upper": {"type": "numeric", "alias": "Upper"},
                    "scale": {"type": "numeric", "alias": "Scale"},
                },
            }

        return sp_scalars_dict, ve_scalars_dict

    def validate_table(
        self, symbol: Set | Parameter | Variable | Equation, last_item
    ):
        if symbol.dimension < 2:
            raise ValidationError(
                "The symbol for miro table must have at least two"
                " domain elements."
            )

        if not isinstance(last_item, (gp.Set, gp.Alias)):
            raise ValidationError(
                "The last domain of the miro table must be a set or an alias"
                f" but found {type(last_item)}"
            )

        if (
            isinstance(symbol.domain_forwarding, list)
            and symbol.domain_forwarding[-1]
        ) or (
            isinstance(symbol.domain_forwarding, bool)
            and symbol.domain_forwarding
        ):
            raise ValidationError(
                "Cannot use domain forwarding feature for miro tables."
            )

        if hasattr(last_item, "_is_miro_input") and last_item._is_miro_input:
            raise ValidationError(
                "The last column of miro table cannot be a miro input!"
            )

    def prepare_headers_dict(
        self, symbol: Set | Parameter | Variable | Equation
    ):
        type_map = {
            str: "string",
            "float64": "numeric",
            "category": "string",
            "object": "string",
        }

        if isinstance(symbol, gp.Set) and symbol.records is None:
            domain_keys = ["uni", "element_text"]
            domain_values = [
                {"type": "string", "alias": "uni"},
                {"type": "string", "alias": "element_text"},
            ]
            return dict(zip(domain_keys, domain_values))

        domain_keys = symbol.records.columns.to_list()
        domain_values = []

        for dtype, column in zip(symbol.records.dtypes, domain_keys):
            try:
                elem = self.container[column]
                alias = elem.description if elem.description else elem.name
            except KeyError:
                alias = column

            domain_values.append(
                {"type": type_map[dtype.name], "alias": alias}
            )

        if isinstance(symbol, gp.Parameter) and symbol._is_miro_table:
            last_item = symbol.domain[-1]
            self.validate_table(symbol, last_item)

            if isinstance(last_item, (gp.Set, gp.Alias)):
                set_values = last_item.records["uni"].values.tolist()

                domain_keys = domain_keys[:-2]
                domain_keys += set_values

                domain_values = domain_values[:-2]
                for elem in last_item.records["uni"].values.tolist():
                    domain_values.append({"type": "numeric", "alias": elem})

        assert len(domain_keys) == len(domain_values)
        return dict(zip(domain_keys, domain_values))

    def prepare_symbols(self, symbols: list[str]) -> list[dict]:
        type_map = {
            gp.Parameter: "parameter",
            gp.Set: "set",
            gp.Variable: "variable",
            gp.Equation: "equation",
        }

        info = []
        for name in symbols:
            symbol = self.container[name]

            headers_dict = self.prepare_headers_dict(symbol)

            info.append({
                "alias": (
                    symbol.description if symbol.description else symbol.name
                ),
                "symtype": type_map[type(symbol)],
                "headers": headers_dict,
            })

        return info

    def prepare_symbols_dict(self, is_input: bool, symbols: list[str]) -> dict:
        alias = "Input Scalars" if is_input else "Output Scalars"
        scalars_sp_dict, scalars_ve_dict = self._prepare_scalars(
            alias, self.input_scalars if is_input else self.output_scalars
        )

        non_scalars = (
            list(set(symbols) - set(self.input_scalars))
            if is_input
            else list(set(symbols) - set(self.output_scalars))
        )
        symbol_dicts = self.prepare_symbols(non_scalars)

        non_scalars = [name for name in non_scalars]

        keys = non_scalars
        values = symbol_dicts

        if scalars_sp_dict is not None:
            keys.append("_scalars" if is_input else "_scalars_out")
            values.append(scalars_sp_dict)

        if scalars_ve_dict is not None:
            keys.append("_scalarsve_out")
            values.append(scalars_ve_dict)

        symbols_dict = dict(zip(keys, values))

        return symbols_dict

    def prepare_json(self) -> str:
        input_symbols_dict = self.prepare_symbols_dict(
            True, self.input_symbols
        )
        output_symbols_dict = self.prepare_symbols_dict(
            False, self.output_symbols
        )

        miro_dict = {
            "modelTitle": self.model_title,
            "inputSymbols": input_symbols_dict,
            "outputSymbols": output_symbols_dict,
        }

        return json.dumps(miro_dict, indent=4)

    def writeJson(self):
        content = self.prepare_json()

        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        directory = os.path.dirname(sys.argv[0])
        conf_path = os.path.join(directory, f"conf_{filename}")
        try:
            os.mkdir(conf_path)
        except FileExistsError:
            pass

        with open(os.path.join(conf_path, f"{filename}_io.json"), "w") as conf:
            conf.write(content)
