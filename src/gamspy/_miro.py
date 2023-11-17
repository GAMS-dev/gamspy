import json
import os
import sys
from typing import List
from typing import TYPE_CHECKING

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
        self.miro_json = self._prepare_json()

    def _prepare_input_scalars(self):
        scalar_names = []
        scalar_texts = []
        scalar_types = []
        for name in self.input_symbols:
            symbol = self.container[name]
            if (isinstance(symbol, gp.Parameter) and symbol.is_scalar) or (
                isinstance(symbol, gp.Set) and symbol.is_singleton
            ):
                scalar_names.append(name)
                scalar_texts.append(symbol.description)

                if isinstance(symbol, gp.Parameter):
                    scalar_types.append("parameter")
                else:
                    scalar_types.append("set")

        if len(scalar_names) == 0:
            return None

        scalars_dict = {
            "alias": "Input Scalars",
            "symnames": scalar_names,
            "symtext": scalar_texts,
            "symtypes": scalar_types,
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

        return scalars_dict

    def _prepare_input_symbols(self):
        type_map = {
            gp.Parameter: "parameter",
            gp.Set: "set",
            str: "string",
            "float64": "numeric",
            "category": "string",
        }

        info = []
        for name in self.input_symbols:
            symbol = self.container[name]
            domain_keys = symbol.records.columns.to_list()
            domain_values = []

            for dtype, column in zip(
                symbol.records.dtypes, symbol.records.columns.to_list()
            ):
                domain_values.append(
                    {"type": type_map[dtype.name], "alias": column}
                )

            headers_dict = dict(zip(domain_keys, domain_values))

            info.append(
                {
                    "alias": symbol.name,
                    "symtype": type_map[type(symbol)],
                    "headers": headers_dict,
                }
            )

        return info

    def _prepare_input_symbols_dict(self):
        scalars_dict = self._prepare_input_scalars()
        symbol_dicts = self._prepare_input_symbols()

        keys = self.input_symbols
        values = symbol_dicts

        if scalars_dict is not None:
            keys.append("_scalars")
            values.append(scalars_dict)

        input_symbols_dict = dict(zip(keys, values))

        return input_symbols_dict

    def _prepare_output_symbols(self):
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
        for name in self.output_symbols:
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
                    "alias": symbol.name,
                    "symtype": type_map[type(symbol)],
                    "headers": headers_dict,
                }
            )

        return info

    def _prepare_output_scalars(self):
        scalar_names = []
        scalar_texts = []
        scalar_types = []
        for name in self.output_symbols:
            symbol = self.container[name]
            if (isinstance(symbol, gp.Parameter) and symbol.is_scalar) or (
                isinstance(symbol, gp.Set) and symbol.is_singleton
            ):
                scalar_names.append(name)
                scalar_texts.append(symbol.description)

                if isinstance(symbol, gp.Parameter):
                    scalar_types.append("parameter")
                else:
                    scalar_types.append("set")

        if len(scalar_names) == 0:
            return None

        scalars_dict = {
            "alias": "Input Scalars",
            "symnames": scalar_names,
            "symtext": scalar_texts,
            "symtypes": scalar_types,
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

        return scalars_dict

    def _prepare_output_symbols_dict(self):
        scalars_dict = self._prepare_output_scalars()
        symbol_dicts = self._prepare_output_symbols()

        keys = self.output_symbols
        values = symbol_dicts

        if scalars_dict is not None:
            keys.append("_scalars_out")
            values.append(scalars_dict)

        output_symbols_dict = dict(zip(keys, values))

        return output_symbols_dict

    def _prepare_json(self) -> str:
        input_symbols_dict = self._prepare_input_symbols_dict()
        output_symbols_dict = self._prepare_output_symbols_dict()

        miro_dict = {
            "modelTitle": self.model_title,
            "inputSymbols": input_symbols_dict,
            "outputSymbols": output_symbols_dict,
        }

        return json.dumps(miro_dict)

    def writeJson(self):
        content = self._prepare_json()

        filename = sys.argv[0].split(".")[0]
        conf_path = f"conf_{filename}"
        try:
            os.mkdir(conf_path)
        except FileExistsError:
            pass

        with open(conf_path + os.sep + f"{filename}_io.json", "w") as conf:
            conf.write(content)
