import json
import os
import sys
from typing import List
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
    ) -> Union[dict, None]:
        names = []
        texts = []
        types = []
        for name in symbols:
            symbol = self.container[name]
            names.append(name)
            texts.append(
                symbol.description if symbol.description else symbol.name
            )
            types.append(type(symbol).__name__.lower())

        if len(names) == 0:
            return None

        scalars_dict = {
            "alias": alias,
            "symnames": names,
            "symtext": texts,
            "symtypes": types,
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
        scalars_dict = self._prepare_scalars(
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

        if scalars_dict is not None:
            keys.append("_scalars" if is_input else "_scalars_out")
            values.append(scalars_dict)

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

        filename = os.path.basename(sys.argv[0]).split(".")[0]
        conf_path = f"conf_{filename}"
        try:
            os.mkdir(conf_path)
        except FileExistsError:
            pass

        with open(conf_path + os.sep + f"{filename}_io.json", "w") as conf:
            conf.write(content)
