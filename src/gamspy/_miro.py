from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import gamspy as gp
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Equation, Parameter, Set, Variable

MIRO_GDX_IN = os.getenv("GAMS_IDC_GDX_INPUT", None)
MIRO_GDX_OUT = os.getenv("GAMS_IDC_GDX_OUTPUT", None)


def get_load_input_str(statement: str, gdx_in: str) -> str:  # pragma: no cover
    string = "$gdxIn\n"  # close the old one
    string += f"$gdxIn {MIRO_GDX_IN}\n"  # open the new one
    string += f"$loadDC {statement}\n"
    string += "$gdxIn\n"  # close the new one
    string += f"$gdxIn {gdx_in}\n"

    return string


def get_unload_output_str(container: Container) -> str:  # pragma: no cover
    unload_str = ",".join(container._miro_output_symbols)
    return f"execute_unload '{MIRO_GDX_OUT}' {unload_str}\n"


def load_miro_symbol_records(
    container: Container, is_implicit: bool = False
):  # pragma: no cover
    # Load records of miro input symbols
    if MIRO_GDX_IN and container._miro_input_symbols:
        names = [
            name
            for name in container._miro_input_symbols
            if not container[name]._already_loaded
        ]
        container._load_records_from_gdx(MIRO_GDX_IN, names)
        for name in names:
            symbol = container[name]
            symbol._already_loaded = True
            if isinstance(symbol, gp.Parameter) and symbol._is_miro_table:
                symbol._records.columns = symbol.domain_names + ["value"]

    # Load records of miro output symbols
    if MIRO_GDX_OUT and container._miro_output_symbols and not is_implicit:
        container._load_records_from_gdx(
            MIRO_GDX_OUT, container._miro_output_symbols
        )


class MiroJSONEncoder:
    def __init__(
        self,
        container: Container,
    ):
        self.container = container
        self.model_title = "GAMSPy App"
        self.input_symbols = container._miro_input_symbols
        self.output_symbols = container._miro_output_symbols
        self.input_scalars = self._find_scalars(self.input_symbols)
        self.output_scalars = self._find_scalars(self.output_symbols)

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
        if isinstance(symbol, gp.Set):
            domain_keys = symbol.domain_names + ["element_text"]
            types = ["string"] * (len(symbol.domain_names) + 1)
        elif isinstance(symbol, gp.Parameter):
            domain_keys = symbol.domain_names + ["value"]
            types = ["string"] * len(symbol.domain_names) + ["numeric"]

            if symbol._is_miro_table:
                last_item = symbol.domain[-1]
                self.validate_table(symbol, last_item)

                set_values = last_item.records["uni"].values.tolist()

                domain_keys = domain_keys[:-2]
                types = ["string"] * len(domain_keys) + ["numeric"] * len(
                    set_values
                )
                domain_keys += set_values
        elif isinstance(symbol, (gp.Variable, gp.Equation)):
            domain_keys = symbol.domain_names + [
                "level",
                "marginal",
                "lower",
                "upper",
                "scale",
            ]
            types = ["string"] * len(symbol.domain_names) + ["numeric"] * 5

        domain_values = []
        uni_counter = 0
        for idx, key in enumerate(domain_keys):
            if key == "*":
                domain_keys[idx] = (
                    "uni" if uni_counter == 0 else f"uni{uni_counter}"
                )
                uni_counter += 1

        for column, column_type in zip(domain_keys, types):
            try:
                elem = self.container[column]
                alias = elem.description if elem.description else column
            except KeyError:
                alias = column

            domain_values.append({"type": column_type, "alias": alias})

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

            info.append(
                {
                    "alias": (
                        symbol.description
                        if symbol.description
                        else symbol.name
                    ),
                    "symtype": type_map[type(symbol)],
                    "headers": headers_dict,
                }
            )

        return info

    def prepare_symbols_dict(self, is_input: bool, symbols: list[str]) -> dict:
        alias = "Input Scalars" if is_input else "Output Scalars"
        scalars_sp_dict, scalars_ve_dict = self._prepare_scalars(
            alias, self.input_scalars if is_input else self.output_scalars
        )

        non_scalars = []

        for symbol in symbols:
            if is_input:
                if symbol not in self.input_scalars:
                    non_scalars.append(symbol)
            else:
                if symbol not in self.output_scalars:
                    non_scalars.append(symbol)

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

    def prepare_dict(self) -> dict:
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

        return miro_dict

    def write_json(self):
        miro_dict = self.prepare_dict()

        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        directory = os.path.dirname(sys.argv[0])
        conf_path = os.path.join(directory, f"conf_{filename}")
        try:
            os.mkdir(conf_path)
        except FileExistsError:
            pass

        with open(os.path.join(conf_path, f"{filename}_io.json"), "w") as conf:
            conf.write(json.dumps(miro_dict, indent=4))

        return miro_dict
