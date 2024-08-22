from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod

import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
from gamspy._options import EXECUTION_OPTIONS, MODEL_ATTR_OPTION_MAP, Options
from gamspy.exceptions import LatexException, ValidationError

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
        all_symbols = get_symbols(self.model)

        all_needed_symbols = sorted(
            all_symbols, key=list(self.container.data.keys()).index
        )

        return all_needed_symbols

    def convert(self, options: Options | None = None):
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
            options.export(os.path.join(self.path, f"{self.model.name}.pf"))
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
            f'GAMS model has been generated under {os.path.join(self.path, self.model.name + ".gms")}'
        )


TABLE_HEADER = """\\begin{tabularx}{\\textwidth}{| l | l | X |}
\\hline
\\textbf{Name} & \\textbf{Domains} & \\textbf{Description}\\\\
\\hline
\\endhead
"""

TABLE_FOOTER = "\\hline\n\\end{tabularx}"


class LatexConverter(Converter):
    def __init__(self, model, path) -> None:
        os.makedirs(path, exist_ok=True)
        self.model = model
        self.container = model.container
        self.path = path
        self.symbols = get_symbols(model)
        self.header = self.get_header()
        self.set_header = "\\subsection*{Sets}"
        self.param_header = "\\subsection*{Parameters}"
        self.variable_header = "\\subsection*{Variables}"
        self.equation_header = "\\subsection*{Equations}"
        self.definitions_header = "\\subsection*{Equation Definitions}"
        self.footer = "\\end{document}"
        self.tex_path = os.path.join(path, model.name + ".tex")

    def convert(self) -> None:
        latex_strs = [self.header]

        # Sets
        latex_strs.append("\\subsection*{Sets}")
        latex_strs.append(self.get_table(syms.Set))

        # Parameters
        latex_strs.append("\\subsection*{Parameters}")
        latex_strs.append(self.get_table(syms.Parameter))

        # Variables
        latex_strs.append("\\subsection*{Variables}")
        latex_strs.append(self.get_table(syms.Variable))

        # Equations
        latex_strs.append("\\subsection*{Equations}")
        latex_strs.append(self.get_table(syms.Equation))

        # Definitions
        latex_strs.append("\\section*{Equation Definitions}")
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
            f'Latex (.tex) file has been generated under {os.path.join(self.path, self.model.name + ".tex")}'
        )

        self.latex_str = latex_str

    def to_pdf(self):
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

    def get_table(self, symbol_type) -> str:
        table = [TABLE_HEADER]
        for name in self.symbols:
            symbol = self.container[name]
            if isinstance(symbol, symbol_type):
                summary = symbol.summary
                row = f'{summary["name"]} & {",".join(summary["domain"])} & {summary["description"]}\\\\'
                row = row.replace("_", "\_")
                table.append(row)

        table.append(TABLE_FOOTER)

        return "\n".join(table)

    def get_definitions(self) -> str:
        definitions = []
        for equation in self.model.equations:
            domain_str = ",".join([elem.name for elem in equation.domain])
            equation_str = "\\subsubsection*{$" + equation.name.replace(
                "_", "\_"
            )
            if domain_str:
                equation_str += f"_{{{domain_str}}}"
            equation_str += "$}\n\\begin{equation}\n"

            equation_str += (
                equation._definition.right.latexRepr() + "\n\\end{equation}\n"
            )

            if isinstance(
                equation._definition.left, implicits.ImplicitEquation
            ):
                if len(equation._definition.left.domain) > 0:
                    domain_str = ",".join(
                        [
                            symbol.name
                            for symbol in equation._definition.left.domain
                        ]
                    )
                    domain_str = f"\\hfill\n$\n\\forall {domain_str}"
                    equation_str += f"{domain_str}\n$"
            else:
                domain_str = ",".join(
                    [
                        symbol.name
                        for symbol in equation._definition.left.conditioning_on.domain
                    ]
                )
                domain_str = f"\\hfill\n$\n\\forall {domain_str}"
                constraint_str = (
                    equation._definition.left.condition.latexRepr()
                )
                equation_str += f"{domain_str} ~ | ~ {constraint_str} \n$"

            equation_str += "\\vspace{5pt}\n\\hrule"
            definitions.append(equation_str)

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
\\usepackage{xcolor}
\\setlength{\parindent}{0pt}

\\begin{document}
\section*{Symbols}

"""
        return header
