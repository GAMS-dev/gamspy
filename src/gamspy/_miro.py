import json
import os
import sys
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gamspy as gp

if TYPE_CHECKING:
    from gamspy import Container


class MiroJSONEncoder:
    def __init__(
        self,
        container: "Container",
        input_symbols: List[str],
        output_symbols: List[str],
    ):
        self.container = container
        self.model_title = "GAMSPy App"
        self.input_symbols = input_symbols
        self.output_symbols = output_symbols
        self.input_scalars = self._find_scalars(input_symbols)
        self.output_scalars = self._find_scalars(output_symbols)
        self.miro_json = self._prepare_json()

    def _find_scalars(self, symbols: List[str]) -> List[str]:
        scalars = []
        for name in symbols:
            symbol = self.container[name]

            if len(symbol.domain) == 0:
                scalars.append(name)

        return scalars

    def _prepare_scalars(
        self, alias: str, symbols: List[str]
    ) -> Tuple[Union[dict, None], Union[dict, None]]:
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

    def _prepare_symbols(self, symbols: List[str]) -> List[dict]:
        type_map = {
            gp.Parameter: "parameter",
            gp.Set: "set",
            gp.Variable: "variable",
            gp.Equation: "equation",
            str: "string",
            "float64": "numeric",
            "category": "string",
        }

        info = []
        for name in symbols:
            symbol = self.container[name]

            domain_keys = symbol.records.columns.to_list()
            domain_values = []

            for dtype, column in zip(symbol.records.dtypes, domain_keys):
                domain_values.append(
                    {"type": type_map[dtype.name], "alias": column}
                )

            headers_dict = dict(zip(domain_keys, domain_values))

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

    def _prepare_symbols_dict(
        self, is_input: bool, symbols: List[str]
    ) -> dict:
        alias = "Input Scalars" if is_input else "Output Scalars"
        scalars_sp_dict, scalars_ve_dict = self._prepare_scalars(
            alias, self.input_scalars if is_input else self.output_scalars
        )

        non_scalars = (
            list(set(symbols) - set(self.input_scalars))
            if is_input
            else list(set(symbols) - set(self.output_scalars))
        )
        symbol_dicts = self._prepare_symbols(non_scalars)

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

    def _prepare_json(self) -> str:
        input_symbols_dict = self._prepare_symbols_dict(
            True, self.input_symbols
        )
        output_symbols_dict = self._prepare_symbols_dict(
            False, self.output_symbols
        )

        miro_dict = {
            "modelTitle": self.model_title,
            "inputSymbols": input_symbols_dict,
            "outputSymbols": output_symbols_dict,
        }

        return json.dumps(miro_dict)

    def writeJson(self):
        content = self._prepare_json()

        filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        directory = os.path.dirname(sys.argv[0])
        conf_path = os.path.join(directory, f"conf_{filename}")
        try:
            os.mkdir(conf_path)
        except FileExistsError:
            pass

        with open(os.path.join(conf_path, f"{filename}_io.json"), "w") as conf:
            conf.write(content)
