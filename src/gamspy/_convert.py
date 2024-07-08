from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import gamspy._symbols as syms

logger = logging.getLogger("CONVERTER")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Converter(ABC):
    @abstractmethod
    def convert(): ...


class GamsConverter(Converter):
    def __init__(self, model, path) -> None:
        os.makedirs(path, exist_ok=True)
        self.model = model
        self.container = model.container
        self.path = path
        self.gdx_path = os.path.join(path, model.name + "_data.gdx")
        self.gms_path = os.path.join(path, model.name + ".gms")

    def get_definitions(self) -> list[str]:
        definitions = []
        for equation in self.model.equations:
            definitions.append(equation._definition.getDeclaration())

        if self.model._matches:
            for equation in self.model._matches:
                definitions.append(equation._definition.getDeclaration())

        return definitions

    def get_all_symbols(self) -> list[str]:
        all_symbols = []
        for equation in self.model.equations:
            symbols = equation._definition._find_all_symbols()

            for symbol in symbols:
                if symbol not in all_symbols:
                    all_symbols.append(symbol)

        if self.model._matches:
            for equation in self.model._matches:
                symbols = equation._definition._find_all_symbols()
                for symbol in symbols:
                    if symbol not in all_symbols:
                        all_symbols.append(symbol)

        all_needed_symbols = sorted(all_symbols, key=self.sort_names)

        return all_needed_symbols

    def convert(self):
        """Generates .gms and .gdx file"""
        symbols = self.get_all_symbols()
        self.container.write(
            self.gdx_path, symbols
        )  # Write the symbol data first

        declarations = [
            self.container[name].getDeclaration() for name in symbols
        ]
        definitions = self.get_definitions()
        load_str = f"$gdxLoadAll {os.path.abspath(self.gdx_path)}"
        declarations.append(self.model.getDeclaration())
        solve_string = self.model._generate_solve_string()
        strings = [*declarations, load_str, *definitions, solve_string]

        gams_string = "\n".join(strings)
        with open(self.gms_path, "w") as file:  # Write the GAMS code
            file.write(gams_string)

        logger.info(
            f'GAMS model has been generated under {os.path.join(self.path, self.model.name + ".gms")}'
        )

    def sort_names(self, name: str) -> int:
        PRECEDENCE = {
            syms.Set: 1,
            syms.Alias: 1,
            syms.Parameter: 3,
            syms.Variable: 4,
            syms.Equation: 5,
        }

        symbol = self.container[name]
        precedence = PRECEDENCE[type(symbol)]

        if isinstance(symbol, syms.Set) and any(
            not isinstance(elem, str) for elem in symbol.domain
        ):
            precedence = 2

        return precedence
