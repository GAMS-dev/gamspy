"""
Notations for the mathematical representation of GAMSPy expressions.

The structure of a representation (operator precedence, parentheses, the scope
of big operators such as sums) is decided by the expression tree itself. A
notation only decides how each part of it is written.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LATEX_OPERATORS = {
    "=g=": "\\geq",
    "=l=": "\\leq",
    "=e=": "=",
    "=n=": "\\text{=n=}",
    "=x=": "\\text{=x=}",
    "=b=": "\\text{=b=}",
    "eq": "=",
    "ne": "\\neq",
    ">=": "\\geq",
    "<=": "\\leq",
    "*": "\\cdot",
    "and": "\\wedge",
    "or": "\\vee",
    "xor": "\\oplus",
    "not": "\\neg ",
    "$=": "\\stackrel{\\$}{=}",
}

LATEX_SPECIAL_VALUES = {
    "EPS": "\\text{EPS}",
    "NA": "\\text{NA}",
    "UNDF": "\\text{UNDF}",
    "INF": "\\infty",
    "-INF": "-\\infty",
}

LATEX_BIG_OPERATORS = {
    "sum": "\\sum",
    "prod": "\\prod",
    "smax": "\\max",
    "smin": "\\min",
    "sand": "\\bigwedge",
    "sor": "\\bigvee",
}

# Functions that have a dedicated LaTeX command. Note that log is the natural logarithm.
LATEX_FUNCTIONS = {
    "exp": "\\exp",
    "log": "\\ln",
    "log10": "\\log_{10}",
    "log2": "\\log_{2}",
    "sin": "\\sin",
    "cos": "\\cos",
    "tan": "\\tan",
    "sinh": "\\sinh",
    "cosh": "\\cosh",
    "tanh": "\\tanh",
    "arcsin": "\\arcsin",
    "arccos": "\\arccos",
    "arctan": "\\arctan",
    "max": "\\max",
    "min": "\\min",
}

LATEX_DELIMITERS = {
    "abs": ("\\lvert ", " \\rvert"),
    "ceil": ("\\lceil ", " \\rceil"),
    "floor": ("\\lfloor ", " \\rfloor"),
}

UNICODE_OPERATORS = {
    "=g=": "≥",
    "=l=": "≤",
    "=e=": "=",
    "eq": "=",
    "ne": "≠",
    ">=": "≥",
    "<=": "≤",
    "*": "·",
    "and": "∧",
    "or": "\N{LOGICAL OR}",
    "xor": "⊕",
    "not": "¬",
}

UNICODE_SPECIAL_VALUES = {
    "EPS": "EPS",
    "NA": "NA",
    "UNDF": "UNDF",
    "INF": "∞",
    "-INF": "-∞",
}

UNICODE_BIG_OPERATORS = {
    "sum": "∑",
    "prod": "∏",
    "smax": "max",
    "smin": "min",
    "sand": "⋀",
    "sor": "\N{N-ARY LOGICAL OR}",
}

# Functions that are not written with their GAMS name. Note that log is the natural logarithm.
UNICODE_FUNCTIONS = {"log": "ln", "log10": "log₁₀", "log2": "log₂"}

UNICODE_DELIMITERS = {"abs": ("|", "|"), "ceil": ("⌈", "⌉"), "floor": ("⌊", "⌋")}

# Unicode has superscript forms of digits but not of a decimal point or most letters.
SUPERSCRIPTABLE = "0123456789-"
SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
UNICODE_SUPERSCRIPTS = str.maketrans(SUPERSCRIPTABLE, SUPERSCRIPTS)


class Notation(ABC):
    """
    Base class of the notations that can represent GAMSPy expressions.

    Parameters
    ----------
    rename : Mapping[str, str], optional
        Maps symbol names to the names that are written instead. Renamed
        names are written as they are, without escaping.
    """

    # Operators of an expression. Operators that are not listed are written as they are.
    operators: Mapping[str, str]
    # GAMS special values.
    special_values: Mapping[str, str]
    # Operations over a domain such as sum and prod.
    big_operators: Mapping[str, str]
    # Functions that are written with a dedicated name (e.g. log -> ln).
    functions: Mapping[str, str]
    # Functions that are written by delimiting their operands (e.g. |x|).
    delimiters: Mapping[str, tuple[str, str]]
    # Separates an expression from the condition that restricts it.
    such_that: str
    # Separates an equation definition from its domain.
    forall: str
    # Separates an equation definition from its condition if it has no domain.
    condition_only: str
    # Separates an equation definition from its matched variables.
    complementarity: str

    def __init__(self, rename: Mapping[str, str] | None = None) -> None:
        self.rename = {} if rename is None else rename

    def symbol_name(self, name: str) -> str:
        """Name of a symbol in this notation."""
        if name in self.rename:
            return self.rename[name]

        return self.escape(name)

    def function_name(self, name: str) -> str:
        """Name of a function in this notation."""
        if name in self.functions:
            return self.functions[name]

        return self.operator_name(name)

    @abstractmethod
    def escape(self, text: str) -> str:
        """Escapes the characters of a text that have a special meaning."""

    @abstractmethod
    def operator_name(self, name: str) -> str:
        """Name of a function or an operator without a dedicated name."""

    @abstractmethod
    def indexed(self, name: str, indices: Sequence[str]) -> str:
        """A symbol that is indexed by the given indices (e.g. x_{i,j})."""

    @abstractmethod
    def attribute(self, name: str, attribute: str) -> str:
        """An attribute of a symbol (e.g. the level of a variable)."""

    @abstractmethod
    def label(self, text: str) -> str:
        """A label in the domain of a symbol."""

    @abstractmethod
    def quoted(self, text: str) -> str:
        """A label that is used as an operand or an index (e.g. x['i1'])."""

    @abstractmethod
    def big_operator(self, op_name: str, index: str, body: str) -> str:
        """An operation over a domain (e.g. sum_{i} x_{i})."""

    @abstractmethod
    def power(self, base: str, exponent: str, is_exponent_leaf: bool) -> str:
        """Raises base to the given exponent."""

    @abstractmethod
    def is_superscripted(self, text: str) -> bool:
        """Whether the text contains a superscript that would clash with a power."""

    @abstractmethod
    def sqrt(self, operand: str) -> str:
        """Square root of the operand."""

    @abstractmethod
    def fraction(self, numerator: str, denominator: str) -> str | None:
        """
        A fraction that groups both of its operands. Returns None if the
        notation writes a division as a binary operator instead.
        """

    @abstractmethod
    def math_block(self, body: str) -> str:
        """Wraps an equation definition."""


class LatexNotation(Notation):
    operators = LATEX_OPERATORS
    special_values = LATEX_SPECIAL_VALUES
    big_operators = LATEX_BIG_OPERATORS
    functions = LATEX_FUNCTIONS
    delimiters = LATEX_DELIMITERS
    such_that = " ~ | ~ "
    forall = "\\hfill \\forall "
    condition_only = "\\hfill \\text{if } "
    complementarity = " \\perp "

    def escape(self, text: str) -> str:
        return text.replace("_", r"\_")

    def operator_name(self, name: str) -> str:
        return f"\\operatorname{{{self.escape(name)}}}"

    def indexed(self, name: str, indices: Sequence[str]) -> str:
        return f"{name}_{{{','.join(indices)}}}"

    def attribute(self, name: str, attribute: str) -> str:
        return f"{{{name}}}^{{\\mathrm{{{attribute}}}}}"

    def label(self, text: str) -> str:
        return f"\\text{{{self.escape(text)}}}"

    def quoted(self, text: str) -> str:
        return f"\\text{{`{self.escape(text)}'}}"

    def big_operator(self, op_name: str, index: str, body: str) -> str:
        return f"{self.big_operators[op_name]}_{{{index}}} {body}"

    def power(self, base: str, exponent: str, is_exponent_leaf: bool) -> str:
        # The braces already group the exponent.
        return f"{base}^{{{exponent}}}"

    def is_superscripted(self, text: str) -> bool:
        return "^" in text

    def sqrt(self, operand: str) -> str:
        return f"\\sqrt{{{operand}}}"

    def fraction(self, numerator: str, denominator: str) -> str:
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    def math_block(self, body: str) -> str:
        return f"$\n{body}\n$"


class UnicodeNotation(Notation):
    operators = UNICODE_OPERATORS
    special_values = UNICODE_SPECIAL_VALUES
    big_operators = UNICODE_BIG_OPERATORS
    functions = UNICODE_FUNCTIONS
    delimiters = UNICODE_DELIMITERS
    such_that = " | "
    forall = "  ∀ "
    condition_only = "  if "
    complementarity = " ⊥ "

    def escape(self, text: str) -> str:
        return text

    def operator_name(self, name: str) -> str:
        return name

    def indexed(self, name: str, indices: Sequence[str]) -> str:
        # Unicode has no subscript form of most letters, so indices are bracketed.
        return f"{name}[{','.join(indices)}]"

    def attribute(self, name: str, attribute: str) -> str:
        return f"{name}.{attribute}"

    def label(self, text: str) -> str:
        return self.quoted(text)

    def quoted(self, text: str) -> str:
        return f"'{text}'"

    def big_operator(self, op_name: str, index: str, body: str) -> str:
        return f"{self.big_operators[op_name]}({index}) {body}"

    def power(self, base: str, exponent: str, is_exponent_leaf: bool) -> str:
        if all(char in SUPERSCRIPTABLE for char in exponent):
            return f"{base}{exponent.translate(UNICODE_SUPERSCRIPTS)}"

        if not is_exponent_leaf:
            exponent = f"({exponent})"

        return f"{base}^{exponent}"

    def is_superscripted(self, text: str) -> bool:
        return "^" in text or any(char in SUPERSCRIPTS for char in text)

    def sqrt(self, operand: str) -> str:
        return f"√({operand})"

    def fraction(self, numerator: str, denominator: str) -> None:
        return None

    def math_block(self, body: str) -> str:
        return body


LATEX = LatexNotation()
UNICODE = UnicodeNotation()
