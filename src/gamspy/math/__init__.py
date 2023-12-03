#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

from gamspy.math.log_power import cv_power
from gamspy.math.log_power import exp
from gamspy.math.log_power import log
from gamspy.math.log_power import log10
from gamspy.math.log_power import log2
from gamspy.math.log_power import log_beta
from gamspy.math.log_power import log_gamma
from gamspy.math.log_power import logit
from gamspy.math.log_power import power
from gamspy.math.log_power import rpower
from gamspy.math.log_power import sign_power
from gamspy.math.log_power import sllog10
from gamspy.math.log_power import sqlog10
from gamspy.math.log_power import sqr
from gamspy.math.log_power import vc_power
from gamspy.math.misc import abs
from gamspy.math.misc import beta
from gamspy.math.misc import bool_and
from gamspy.math.misc import bool_eqv
from gamspy.math.misc import bool_imp
from gamspy.math.misc import bool_not
from gamspy.math.misc import bool_or
from gamspy.math.misc import bool_xor
from gamspy.math.misc import ceil
from gamspy.math.misc import dist
from gamspy.math.misc import div
from gamspy.math.misc import div0
from gamspy.math.misc import entropy
from gamspy.math.misc import errorf
from gamspy.math.misc import factorial
from gamspy.math.misc import floor
from gamspy.math.misc import fractional
from gamspy.math.misc import gamma
from gamspy.math.misc import ifthen
from gamspy.math.misc import lse_max
from gamspy.math.misc import lse_max_sc
from gamspy.math.misc import lse_min
from gamspy.math.misc import lse_min_sc
from gamspy.math.misc import Max
from gamspy.math.misc import Min
from gamspy.math.misc import mod
from gamspy.math.misc import ncp_cm
from gamspy.math.misc import ncp_f
from gamspy.math.misc import ncpVUpow
from gamspy.math.misc import ncpVUsin
from gamspy.math.misc import poly
from gamspy.math.misc import rand_binomial
from gamspy.math.misc import rand_linear
from gamspy.math.misc import rand_triangle
from gamspy.math.misc import regularized_beta
from gamspy.math.misc import regularized_gamma
from gamspy.math.misc import rel_eq
from gamspy.math.misc import rel_ge
from gamspy.math.misc import rel_gt
from gamspy.math.misc import rel_le
from gamspy.math.misc import rel_lt
from gamspy.math.misc import rel_ne
from gamspy.math.misc import Round
from gamspy.math.misc import sigmoid
from gamspy.math.misc import sign
from gamspy.math.misc import slexp
from gamspy.math.misc import slrec
from gamspy.math.misc import sqexp
from gamspy.math.misc import sqrec
from gamspy.math.misc import sqrt
from gamspy.math.misc import truncate
from gamspy.math.probability import binomial
from gamspy.math.probability import centropy
from gamspy.math.probability import normal
from gamspy.math.probability import uniform
from gamspy.math.probability import uniformInt
from gamspy.math.trigonometric import acos
from gamspy.math.trigonometric import asin
from gamspy.math.trigonometric import atan
from gamspy.math.trigonometric import atan2
from gamspy.math.trigonometric import cos
from gamspy.math.trigonometric import cosh
from gamspy.math.trigonometric import sin
from gamspy.math.trigonometric import sinh
from gamspy.math.trigonometric import tan
from gamspy.math.trigonometric import tanh

__all__ = [
    "cos",
    "cosh",
    "sin",
    "sinh",
    "acos",
    "asin",
    "tan",
    "tanh",
    "atan",
    "atan2",
    "exp",
    "power",
    "sqr",
    "sqrt",
    "log",
    "log2",
    "log10",
    "log_beta",
    "log_gamma",
    "logit",
    "abs",
    "ceil",
    "dist",
    "div",
    "div0",
    "factorial",
    "floor",
    "fractional",
    "Min",
    "Max",
    "mod",
    "Round",
    "sign",
    "binomial",
    "centropy",
    "normal",
    "uniform",
    "uniformInt",
    "cv_power",
    "rpower",
    "sign_power",
    "sllog10",
    "sqlog10",
    "vc_power",
    "slexp",
    "sqexp",
    "truncate",
    "ifthen",
    "beta",
    "regularized_beta",
    "gamma",
    "regularized_gamma",
    "entropy",
    "lse_max",
    "lse_max_sc",
    "lse_min",
    "lse_min_sc",
    "ncp_cm",
    "ncp_f",
    "ncpVUpow",
    "ncpVUsin",
    "poly",
    "rand_binomial",
    "rand_linear",
    "rand_triangle",
    "slrec",
    "sqrec",
    "errorf",
    "sigmoid",
    "bool_and",
    "bool_eqv",
    "bool_imp",
    "bool_not",
    "bool_or",
    "bool_xor",
    "rel_eq",
    "rel_ge",
    "rel_gt",
    "rel_le",
    "rel_lt",
    "rel_ne",
]
