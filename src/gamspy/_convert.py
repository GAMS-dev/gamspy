from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gamspy._symbols as syms
import gamspy.utils as utils
from gamspy._options import EXECUTION_OPTIONS, MODEL_ATTR_OPTION_MAP, Options
from gamspy.exceptions import LatexException, ValidationError

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from gamspy import Alias, Equation, Model, Parameter, Set, Variable

    SymbolType: TypeAlias = Alias | Set | Parameter | Variable | Equation

logger = logging.getLogger("CONVERTER")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_symbols(model) -> list[str]:
    all_symbols = []
    for equation in model.equations:
        symbols = equation._definition._find_all_symbols()

        for symbol in symbols:
            if symbol not in all_symbols:
                all_symbols.append(symbol)

    if model._matches:
        for equation in model._matches:
            symbols = equation._definition._find_all_symbols()
            for symbol in symbols:
                if symbol not in all_symbols:
                    all_symbols.append(symbol)

    return all_symbols


class Converter(ABC):
    @abstractmethod
    def convert(self): ...


class GamsConverter(Converter):
    def __init__(self, model: Model, path: str) -> None:
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
        all_symbols = get_symbols(self.model)

        all_needed_symbols = sorted(
            all_symbols, key=list(self.container.data.keys()).index
        )

        return all_needed_symbols

    def convert(self, options: Options | None = None) -> None:
        """Generates .gms and .gdx file"""
        symbols = self.get_all_symbols()

        # Write the symbol data first
        self.container.write(self.gdx_path, symbols)

        # 1. Declarations
        declarations = [
            self.container[name].getDeclaration() for name in symbols
        ]

        if self.model.external_module is not None:
            em_name = self.model._external_module
            em_file = self.model._external_module_file
            declarations.append(f"File {em_file} /{em_name}/;")
            logger.info("Converter will not copy external module files")
            logger.info(
                f"You need to ensure your external module is accessible from {self.path}"
            )

        # 2. Load the data from gdx
        load_str = f"$gdxLoadAll {os.path.abspath(self.gdx_path)}"

        # 3. Definitions
        definitions = self.get_definitions()
        declarations.append(self.model.getDeclaration())

        # 4. Model attribute options
        options_strs = []
        if options is not None:
            options._export(os.path.join(self.path, f"{self.model.name}.pf"))
            for key, value in options.model_dump(exclude_none=True).items():
                if key in MODEL_ATTR_OPTION_MAP:
                    if isinstance(value, bool):
                        value = int(value)
                    elif isinstance(value, str):
                        value = f"'{value}'"

                    options_strs.append(
                        f"{self.model.name}.{MODEL_ATTR_OPTION_MAP[key]} = {value};"
                    )
                elif key in EXECUTION_OPTIONS:
                    options_strs.append(f"{EXECUTION_OPTIONS[key]} '{value}'")

        # 5. Solve string
        solve_string = self.model._generate_solve_string()
        strings = [
            *declarations,
            load_str,
            *definitions,
            *options_strs,
            solve_string,
        ]

        # Write the GAMS code
        gams_string = "\n".join(strings)
        with open(self.gms_path, "w", encoding="utf-8") as file:
            file.write(gams_string)

        logger.info(
            f'GAMS (.gms) file has been generated under {os.path.join(self.path, self.model.name + ".gms")}'
        )


TABLE_HEADER = """\\begin{tabularx}{\\textwidth}{| l | l | X |}
\\hline
\\textbf{Name} & \\textbf{Domains} & \\textbf{Description}\\\\
\\hline
\\endhead
"""

TABLE_FOOTER = "\\hline\n\\end{tabularx}"


class LatexConverter(Converter):
    def __init__(self, model: Model, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model = model
        self.container = model.container
        self.path = path
        symbols = get_symbols(model)
        if self.model._limited_variables:
            for var in self.model._limited_variables:
                domain = var.domain
                for elem in domain:
                    if (
                        isinstance(elem, (syms.Set, syms.Alias))
                        and elem.name not in symbols
                    ):
                        symbols.append(elem.name)

        self.symbols = sorted(
            symbols, key=list(self.container.data.keys()).index
        )

        for name in self.model._autogen_symbols:
            if name in self.symbols:
                self.symbols.remove(name)

        self.header = self.get_header()
        self.set_header = "\\subsection*{Sets}"
        self.param_header = "\\subsection*{Parameters}"
        self.variable_header = "\\subsection*{Variables}"
        self.equation_header = "\\subsection*{Equations}"
        self.definitions_header = "\\subsection*{Model Definition}"
        self.footer = "\\end{document}"
        self.tex_path = os.path.join(path, model.name + ".tex")

    def convert(self) -> None:
        latex_strs = [self.header]

        # Sets
        latex_strs.append(self.set_header)
        latex_strs.append(self.get_table((syms.Set, syms.Alias)))

        # Parameters
        latex_strs.append(self.param_header)
        latex_strs.append(self.get_table(syms.Parameter))

        # Variables
        latex_strs.append(self.variable_header)
        latex_strs.append(self.get_table(syms.Variable))

        # Equations
        latex_strs.append(self.equation_header)
        latex_strs.append(self.get_table(syms.Equation))

        # Definitions
        latex_strs.append(self.definitions_header)
        if self.model._objective is None:
            ...
        elif isinstance(self.model._objective, syms.Variable):
            latex_strs.append(
                f"\\textbf{{{str(self.model.sense).lower()}}} ${self.model._objective_variable.name}$\\\\"
            )
            latex_strs.append("\\textbf{s.t.}")
        else:
            latex_strs.append(
                f"\\textbf{{{str(self.model.sense).lower()}}} ${self.model._objective.latexRepr()}$\\\\"
            )
            latex_strs.append("\\textbf{s.t.}")

        latex_strs.append(self.get_definitions())

        # Constraints
        latex_strs.append(self.get_constraints())

        latex_strs.append(self.footer)

        latex_str = "\n".join(latex_strs)
        with open(
            self.tex_path, "w", encoding="utf-8"
        ) as file:  # Write the TEX file
            file.write(latex_str)

        logger.info(
            f'LaTeX (.tex) file has been generated under {os.path.join(self.path, self.model.name + ".tex")}'
        )

        self.latex_str = latex_str

    def to_pdf(self) -> None:
        process = subprocess.run(["pdflatex", "-v"])
        if process.returncode:
            raise ValidationError(
                "`pdflatex` is required to generate the pdf! Please install `pdflatex` and add it to the path."
            )

        process = subprocess.run(
            ["pdflatex", f"-output-directory={self.path}", self.tex_path],
            capture_output=True,
            text=True,
        )
        if process.returncode:
            raise LatexException(
                f"Could not generate pdf file: {process.stderr}"
            )

    def get_table(
        self, symbol_type: tuple[Set, Alias] | Parameter | Variable | Equation
    ) -> str:
        table = [TABLE_HEADER]
        for name in self.symbols:
            symbol: SymbolType = self.container[name]

            if isinstance(symbol, symbol_type):
                summary = symbol.summary
                domain_str = ",".join(summary["domain"])
                if (
                    isinstance(symbol, syms.Variable)
                    and self.model._limited_variables
                ):
                    for elem in self.model._limited_variables:
                        if elem.name == symbol.name:
                            domain_str = utils._get_domain_str(elem.domain)[
                                1:-1
                            ]

                row = f'{summary["name"]} & {domain_str} & {summary["description"]}\\\\'
                row = row.replace("_", "\\_")
                table.append(row)

        table.append(TABLE_FOOTER)

        return "\n".join(table)

    def get_definitions(self) -> str:
        definitions = []
        for equation in self.model.equations:
            if equation.name in self.model._autogen_symbols:
                continue

            domain_str = ",".join([elem.name for elem in equation.domain])
            header = "\\subsubsection*{$" + equation.name.replace("_", "\\_")
            if domain_str:
                header += f"_{{{domain_str}}}"
            header += "$}\n"

            footer = "\n\\vspace{5pt}\n\\hrule"
            latex_repr = f"{header}{equation.latexRepr()}{footer}"
            definitions.append(latex_repr)

        return "\n".join(definitions)

    def get_constraints(self) -> str:
        constraints = ["\\bigskip"]
        for name in self.symbols:
            symbol = self.container[name]
            if not isinstance(symbol, syms.Variable):
                continue

            constraint = "$" + symbol.latexRepr()
            if symbol.type == "binary":
                constraint += (
                    "\\in "
                    + r"\{0,1\}"
                    + " ~ \\forall "
                    + ",".join(symbol.domain_names)
                )
            elif symbol.type == "integer":
                constraint += "\\in \\mathbb{Z}_{+} \\forall " + ",".join(
                    symbol.domain_names
                )
            elif symbol.type == "positive":
                constraint += "\\geq 0 ~ \\forall " + ",".join(
                    symbol.domain_names
                )
            elif symbol.type == "negative":
                constraint += "\\leq 0 ~ \\forall " + ",".join(
                    symbol.domain_names
                )
            elif symbol.type == "sos1":
                constraint += "SOS1"
            elif symbol.type == "sos2":
                constraint += "SOS2"
            elif symbol.type == "semicont":
                constraint += "SemiCont"
            elif symbol.type == "semiint":
                constraint += r"\{0, 1, 2, ... \}" + " SemiInt"
            else:
                continue

            constraint += "\\\\$"
            constraints.append(constraint)

        return "\n".join(constraints)

    def get_header(self) -> str:
        header = """\\documentclass[11pt]{article}
\\usepackage{geometry}
\\usepackage[american]{babel}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage[hidelinks]{hyperref}
\\usepackage{tabularx}
\\usepackage{ltablex}
\\keepXColumns

\\begin{document}
\\section*{Symbols}

"""
        return header
