from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import gamspy._symbols as syms
import gamspy.utils as utils
from gamspy._options import EXECUTION_OPTIONS, MODEL_ATTR_OPTION_MAP, Options
from gamspy.exceptions import LatexException, ValidationError

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from gamspy import (
        Alias,
        ConvertOptions,
        Equation,
        FileFormat,
        Model,
        Parameter,
        Set,
        Variable,
    )

    SymbolType: TypeAlias = Union[Alias, Set, Parameter, Variable, Equation]

logger = logging.getLogger("CONVERTER")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

GAMS_JACOBIAN_TEMPLATE = """$onEmpty
Set
   i        'equations index'
   j        'variable index'
   ij(i,j)  'non-zero map'
   s(s)     'sos sets'                        / /
   jc(j)    'continuous variables index'      / /
   jb(j)    'binary variables index'          / /
   ji(j)    'integer variables index'         / /
   jsc(j)   'semi-continuous variables index' / /
   jsi(j)   'semi-integer variables index'    / /
   js1(s,j) 'SOS 1 variables index'           / /
   js2(s,j) 'SOS 2 variables index'           / /
   js(j)    'SOS variables index'
   jd(j)    'discrete variables index'
Singleton set
   jobj(j) 'Objective variable index';

Scalar objcoef 'Min or Max';

Equation e(i) 'Equation';

Variable          xc(j)    'all variables';
Binary variable   xb(j)    'binary variables';
Integer variable  xi(j)    'integer variables';
SemiCont variable xsc(j)   'semi-continuous variables';
SemiInt variable  xsi(j)   'semi-integer variables';
SOS1 variable     xs1(s,j) 'SOS1 variables';
SOS2 variable     xs2(s,j) 'SOS2 variables';

Parameter A(i,j) 'Jacobian';

{}
$gdxin %jacfile%
$load i j jobj objcoef e xc=x A
$if gdxSymExist s $loadM s jb ji jsc jsi js1 js2
$gdxIn

* Extract jc (continuous variables index)
jd(j) = jb(j) or ji(j) or jsc(j) or jsi(j);
option js<js1; jd(js) = yes;
option js<js2; jd(js) = yes;
jc(j) = yes;
jc(jd) = no;

xb.lo(jb)       =xc.lo(jb);  xb.up(jb)       =xc.up(jb);  xb.prior(jb)       =xc.scale(jb); 
xi.lo(ji)       =xc.lo(ji);  xi.up(ji)       =xc.up(ji);  xi.prior(ji)       =xc.scale(ji); 
xsc.lo(jsc)     =xc.lo(jsc); xsc.up(jsc)     =xc.up(jsc); xsc.prior(jsc)     =xc.scale(jsc); 
xsi.lo(jsi)     =xc.lo(jsi); xsi.up(jsi)     =xc.up(jsi); xsi.prior(jsi)     =xc.scale(jsi); 
xs1.lo(js1(s,j))=xc.lo(j);   xs1.up(js1(s,j))=xc.up(j);   xs1.prior(js1(s,j))=xc.scale(j); 
xs2.lo(js2(s,j))=xc.lo(j);   xs2.up(js2(s,j))=xc.up(j);   xs2.prior(js2(s,j))=xc.scale(j);

option ij<A;

Variable xslack(i);
xslack.fx(i) = 0;
xslack.lo(i)$(mapval(e.lo(i))=7) = -inf;
xslack.up(i)$(mapval(e.up(i))=6) =  inf;

Parameter
   rhs(i)  'right hand side'
   AObj(i) 'objective coefficient in row i';

* Replace old objective variable jobj by new one z
AObj(i)   = A(i,jobj);
A(i,jobj) = 0;

* Extract right hand side from equation bounds
rhs(i)$(mapval(e.up(i))=0) = e.up(i);
rhs(i)$(mapval(e.lo(i))=0) = e.lo(i);

* Model
Equation defi(i);
Variable z;

defi(i).. sum(ij(i,j),
               sum(jc(j),    A(i,jc )*xc (jc ))
             + sum(jb(j),    A(i,jb )*xb (jb ))
             + sum(ji(j),    A(i,ji )*xi (ji ))
             + sum(jsc(j),   A(i,jsc)*xsc(jsc))
             + sum(jsi(j),   A(i,jsi)*xsi(jsi))
             + sum(js1(s,j), A(i,j)*xs1(js1))
             + sum(js2(s,j), A(i,j)*xs2(js2)) )
             + z*AObj(i)
               =e= rhs(i) + xslack(i);

Model fromJacobian / defi /;

defi.stage(i) = e.stage(i);

option clear=e, limRow=0, limCol=0, solPrint=off;
fromJacobian.dictFile = 0;
if (card(jd),
   if (objcoef<0,
      solve fromJacobian max z using mip;
   else
      solve fromJacobian min z using mip;
   )
else
   if (objcoef<0,
      solve fromJacobian max z using rmip;
   else
      solve fromJacobian min z using rmip;
   )
)

"""

GAMSPY_JACOBIAN = r"""import sys

from gamspy import (
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    SpecialValues,
    Sum,
    Variable,
    VariableType,
)
from gamspy.math import (
    map_value,
)

if len(sys.argv) != 2:
    print(f"Usage {sys.argv[0]} <jacFile.gdx>")
    exit()

m = Container()

i = Set(m, "i", description="equations index")
j = Set(m, "j", description="variable index")
ij = Set(m, domain=[i, j], description="non-zero map")
s = Set(m, "s", description="sos sets")
jc = Set(m, domain=j, description="continuous variables index")
jb = Set(m, domain=j, description="binary variables index")
ji = Set(m, domain=j, description="integer variables index")
jsc = Set(m, domain=j, description="semi-continuous variables index")
jsi = Set(m, domain=j, description="semi-integer variables index")
js1 = Set(m, domain=[s, j], description="SOS 1 variables index")
js2 = Set(m, domain=[s, j], description="SOS 2 variables index")
js = Set(m, domain=j, description="SOS variables index")
jd = Set(m, domain=j, description="discrete variables index")
jobj = Set(m, domain=j, is_singleton=True, description="Objective variable index")

objcoef = Parameter(m, description="Min or Max")

e = Equation(m, domain=i)

xc = Variable(m, domain=j, type=VariableType.FREE, description="continuous variables")
xb = Variable(m, domain=j, type=VariableType.BINARY, description="binary variables")
xi = Variable(m, domain=j, type=VariableType.INTEGER, description="integer variables")
xsc = Variable(m, domain=j, type=VariableType.SEMICONT, description="semi-continuous variables")
xsi = Variable(m, domain=j, type=VariableType.SEMIINT, description="semi-integer variables")
xs1 = Variable(m, domain=[s, j], type=VariableType.SOS1, description="SOS1 variables")
xs2 = Variable(m, domain=[s, j], type=VariableType.SOS2, description="SOS2 variables")

A = Parameter(m, domain=[i, j], description="Jacobian")

with Container(load_from=sys.argv[1]) as md:
    for sym in ["i", "j", "jobj", "objcoef", "e", "A"]:
        globals()[sym].records = md[sym].records
    xc.records = md["x"].records
    try:
        for sym in ["s", "jb", "ji", "jsc", "jsi", "js1", "js2"]:
            globals()[sym].records = md[sym].records
        mtype = Problem.MIP
    except KeyError:
        mtype = Problem.RMIP

# Extract jc (continuous variables index)
jd[j] = jb[j] | ji[j] | jsc[j] | jsi[j]
js[j] = Sum(js1[s, j], True)
jd[js] = True
js[j] = Sum(js2[s, j], True)
jd[js] = True
jc[j] = True
jc[jd] = False

xb.lo[jb] = xc.lo[jb]
xb.up[jb] = xc.up[jb]
xb.prior[jb] = xc.scale[jb]

xi.lo[ji] = xc.lo[ji]
xi.up[ji] = xc.up[ji]
xi.prior[ji] = xc.scale[ji]

xsc.lo[jsc] = xc.lo[jsc]
xsc.up[jsc] = xc.up[jsc]
xsc.prior[jsc] = xc.scale[jsc]

xsi.lo[jsi] = xc.lo[jsi]
xsi.up[jsi] = xc.up[jsi]
xsi.prior[jsi] = xc.scale[jsi]

xs1.lo[js1[s, j]] = xc.lo[j]
xs1.up[js1[s, j]] = xc.up[j]
xs1.prior[js1[s, j]] = xc.scale[j]

xs2.lo[js1[s, j]] = xc.lo[j]
xs2.up[js1[s, j]] = xc.up[j]
xs2.prior[js1[s, j]] = xc.scale[j]

ij[i, j] = A[i, j]

xslack = Variable(m, domain=i, description="slack variable to make all equations ==")
xslack.fx[i] = 0
xslack.lo[i].where[map_value(e.lo[i]) == 7] = SpecialValues.NEGINF
xslack.up[i].where[map_value(e.up[i]) == 6] = SpecialValues.POSINF

rhs = Parameter(m, domain=i, description="right hand side")
AObj = Parameter(m, domain=i, description="objective coefficient in row i")

# Replace old objective variable jobj by new one z
AObj[i] = A[i, jobj]
A[i, jobj] = 0

# Extract right hand side from equation bounds
rhs[i].where[map_value(e.up[i]) == 0] = e.up[i]
rhs[i].where[map_value(e.lo[i]) == 0] = e.lo[i]

# The model
defi = Equation(m, domain=i)
z = Variable(m)

defi.stage[i] = e.stage[i]
defi[i] = (
    Sum(
        ij[i, j],
        Sum(jc[j], A[i, jc] * xc[jc])
        + Sum(jb[j], A[i, jb] * xb[jb])
        + Sum(ji[j], A[i, ji] * xi[ji])
        + Sum(jsc[j], A[i, jsc] * xsc[jsc])
        + Sum(jsi[j], A[i, jsi] * xsi[jsi])
        + Sum(js1[s, j], A[i, j] * xs1[js1])
        + Sum(js2[s, j], A[i, j] * xs2[js2]),
    )
    + z * AObj[i]
    == rhs[i] + xslack[i]
)

jac = Model(
    m,
    equations=[defi],
    problem=mtype,
    sense=Sense.MAX if objcoef.toValue() < 0 else Sense.MIN,
    objective=z,
)

options = Options(
    variable_listing_limit=0,
    equation_listing_limit=0,
    report_solution=1,
    generate_name_dict=False,
)

jac.solve(output=sys.stdout, options=options)
"""


def get_convert_solver_options(
    path: str,
    file_format: FileFormat | Sequence[FileFormat],
    options: ConvertOptions | None,
) -> dict[str, str]:
    from gamspy._model import FileFormat

    # Key is the GAMSPy option name, value is the GAMS option name.
    FORMAT_RENAME_MAP = {
        "GDXJacobian": "DumpGDX",
        "GDXIntervalEval": "IntervalEval",
    }
    OPTION_RENAME_MAP = {"GAMSInsert": "GmsInsert", "GAMSObjVar": "ObjVar"}

    if not isinstance(file_format, Sequence):
        file_format = [file_format]

    if any(not isinstance(format, FileFormat) for format in file_format):
        raise ValidationError("`file_format` must be a FileFormat enum.")

    solver_options = {}
    for format in file_format:
        name, value = format.name, format.value
        if name in FORMAT_RENAME_MAP:
            name = FORMAT_RENAME_MAP[name]

        if format == FileFormat.GAMSJacobian:
            if FileFormat.GDXJacobian in file_format:
                jacobian_gms = GAMS_JACOBIAN_TEMPLATE.format(
                    f"$if not set jacfile $set jacfile {FileFormat.GDXJacobian.value}"  # type: ignore
                )
            else:
                jacobian_gms = GAMS_JACOBIAN_TEMPLATE.format(
                    "$if not set jacfile $abort Please set --jacfile=<filename>.gdx"
                )

            with open(os.path.join(path, value), "w") as file:
                file.write(jacobian_gms)
        elif format == FileFormat.GAMSPyJacobian:
            with open(os.path.join(path, value), "w") as file:
                file.write(GAMSPY_JACOBIAN)
        else:
            solver_options[name] = os.path.join(path, value)

    if options is not None:
        extra_options = options.model_dump(exclude_none=True)
        for key, value in extra_options.items():
            name = OPTION_RENAME_MAP.get(key, key)
            solver_options[name] = (
                int(value) if isinstance(value, bool) else value
            )

    return solver_options


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


class GamsConverter:
    def __init__(
        self,
        model: Model,
        path: str,
        options: Options | None,
        dump_gams_state: bool,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        self.model = model
        self.container = model.container
        self.path = path
        self.options = options
        self.dump_gams_state = dump_gams_state
        self.gdx_path = os.path.join(path, model.name + "_data.gdx")
        self.gms_path = os.path.join(path, model.name + ".gms")
        self.g00_path = os.path.join(path, model.name + ".g00")

    def get_definitions(self) -> list[str]:
        definitions = []
        for equation in self.model.equations:
            assert equation._definition is not None
            definitions.append(equation._definition.getDeclaration())

        if self.model._matches:
            for key in self.model._matches:
                if isinstance(key, syms.Equation):
                    assert key._definition is not None
                    definitions.append(key._definition.getDeclaration())
                else:
                    for equation in key:
                        assert equation._definition is not None
                        definitions.append(
                            equation._definition.getDeclaration()
                        )

        return definitions

    def get_all_symbols(self) -> list[str]:
        all_symbols = get_symbols(self.model)

        all_needed_symbols = sorted(
            all_symbols, key=list(self.container.data.keys()).index
        )

        return all_needed_symbols

    def convert(self) -> None:
        """Generates .gms, .gdx and .g00 file"""
        symbols = self.get_all_symbols()

        # Write .g00 file
        if self.dump_gams_state:
            self.container._options._set_debug_options({"save": self.g00_path})
            self.container._synch_with_gams()

        # Write .gdx file
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
        load_str = f"$onMultiR\n$gdxLoadAll {os.path.abspath(self.gdx_path)}\n$offMulti"

        # 3. Definitions
        definitions = self.get_definitions()
        declarations.append(self.model.getDeclaration())

        # 4. Model attribute options
        options_strs = []
        if self.options is not None:
            self.options._export(
                os.path.join(self.path, f"{self.model.name}.pf")
            )
            for key, value in self.options.model_dump(
                exclude_none=True
            ).items():
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
        solve_string = self.model._generate_solve_string() + ";"
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
            f"GAMS (.gms) file has been generated under {os.path.join(self.path, self.model.name + '.gms')}"
        )


TABLE_HEADER = """\\begin{tabularx}{\\textwidth}{| l | l | X |}
\\hline
\\textbf{Name} & \\textbf{Domains} & \\textbf{Description}\\\\
\\hline
\\endhead
"""

TABLE_FOOTER = "\\hline\n\\end{tabularx}"


class LatexConverter:
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
                f"\\textbf{{{str(self.model.sense).lower()}}} ${self.model._objective_variable.name}$\\\\"  # type: ignore
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
            f"LaTeX (.tex) file has been generated under {os.path.join(self.path, self.model.name + '.tex')}"
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

    def get_table(self, symbol_type) -> str:
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

                row = f"{summary['name']} & {domain_str} & {summary['description']}\\\\"
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
