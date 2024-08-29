from __future__ import annotations

import argparse
import doctest
import glob
import json
import os
import shutil
import unittest

import gamspy
from integration.test_cmd_script import cmd_suite
from integration.test_engine import engine_suite
from integration.test_external_module import external_module_suite
from integration.test_gamspy_to_gams import gamspy_to_gams_suite
from integration.test_gamspy_to_latex import gamspy_to_latex_suite
from integration.test_miro import miro_suite
from integration.test_model_instance import model_instance_suite
from integration.test_models import gams_models_suite
from integration.test_neos import neos_suite
from integration.test_solve import solve_suite
from unit.test_alias import alias_suite
from unit.test_condition import condition_suite
from unit.test_container import container_suite
from unit.test_domain import domain_suite
from unit.test_equation import equation_suite
from unit.test_extrinsic import extrinsic_suite
from unit.test_magics import magics_suite
from unit.test_math import math_suite
from unit.test_matrix import matrix_suite
from unit.test_model import model_suite
from unit.test_operation import operation_suite
from unit.test_options import options_suite
from unit.test_parameter import parameter_suite
from unit.test_set import set_suite
from unit.test_special_values import special_values_suite
from unit.test_utils import utils_suite
from unit.test_variable import variable_suite

try:
    from dotenv import load_dotenv

    load_dotenv(os.getcwd() + os.sep + ".env")
except Exception:
    pass


class GamspySuite(unittest.TestCase):
    def test_version(self):
        import gamspy

        self.assertEqual(gamspy.__version__, "1.0.0rc1")


class DocsSuite(unittest.TestCase):
    def test_switcher(self):
        this = os.path.dirname(os.path.abspath(__file__))
        root = this.rsplit(os.sep, maxsplit=1)[0]
        with open(
            f"{root}{os.sep}docs{os.sep}_static{os.sep}switcher.json"
        ) as file:
            switcher = json.loads(file.read())
            versions = [elem["version"] for elem in switcher]
            self.assertTrue(f"v{gamspy.__version__}" in versions)

    def test_docs(self):
        root = gamspy.__path__[0]

        api_files = [
            f"{root}{os.sep}_container.py",
            f"{root}{os.sep}_model_instance.py",
            f"{root}{os.sep}_model.py",
            f"{root}{os.sep}utils.py",
            f"{root}{os.sep}_algebra{os.sep}expression.py",
            f"{root}{os.sep}_algebra{os.sep}operation.py",
            f"{root}{os.sep}_algebra{os.sep}domain.py",
            f"{root}{os.sep}_algebra{os.sep}number.py",
            f"{root}{os.sep}_symbols{os.sep}symbol.py",
            f"{root}{os.sep}_symbols{os.sep}alias.py",
            f"{root}{os.sep}_symbols{os.sep}equation.py",
            f"{root}{os.sep}_symbols{os.sep}parameter.py",
            f"{root}{os.sep}_symbols{os.sep}set.py",
            f"{root}{os.sep}_symbols{os.sep}universe_alias.py",
            f"{root}{os.sep}_symbols{os.sep}variable.py",
            f"{root}{os.sep}math{os.sep}matrix.py",
            f"{root}{os.sep}math{os.sep}log_power.py",
            f"{root}{os.sep}math{os.sep}misc.py",
            f"{root}{os.sep}math{os.sep}probability.py",
            f"{root}{os.sep}math{os.sep}trigonometric.py",
            f"{root}{os.sep}math{os.sep}activation.py",
        ]

        for file in api_files:
            results = doctest.testfile(
                file, verbose=True, module_relative=False
            )

            self.assertEqual(results.failed, 0)


def gamspy_suite():
    suite = unittest.TestSuite()
    tests = [
        GamspySuite(name)
        for name in dir(GamspySuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


def docs_suite():
    suite = unittest.TestSuite()
    tests = [
        DocsSuite(name) for name in dir(DocsSuite) if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


def run_integration_tests(
    args: argparse.Namespace, runner: unittest.TextTestRunner
):
    integration_suites = [
        gamspy_to_gams_suite,
        gamspy_to_latex_suite,
        solve_suite,
        model_instance_suite,
        cmd_suite,
        miro_suite,
        external_module_suite,
    ]

    if args.engine:
        integration_suites.append(engine_suite)

    if args.neos:
        integration_suites.append(neos_suite)

    if args.model_library:
        integration_suites.append(gams_models_suite)

    print(f"Running integration tests\n{'='*80}")
    for suite in integration_suites:
        print("=" * 80)
        print(f"\nRunning {suite.__name__}...")
        result = runner.run(suite())
        if not result.wasSuccessful():
            exit(1)
        print("=" * 80)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--doc", action="store_true")
    parser.add_argument("--neos", action="store_true")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--model-library", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs("tmp", exist_ok=True)

    runner = unittest.TextTestRunner()

    unittest_suites = [
        gamspy_suite,
        container_suite,
        math_suite,
        matrix_suite,
        utils_suite,
        set_suite,
        alias_suite,
        parameter_suite,
        variable_suite,
        equation_suite,
        model_suite,
        operation_suite,
        domain_suite,
        condition_suite,
        magics_suite,
        options_suite,
        special_values_suite,
        extrinsic_suite,
    ]

    print(f"Running unittests\n{'='*80}")
    for suite in unittest_suites:
        print("=" * 80)
        print(f"\nRunning {suite.__name__}...")
        result = runner.run(suite())
        if not result.wasSuccessful():
            return 1
        print("=" * 80)

    if args.doc:
        print("=" * 80)
        print(f"\nRunning {suite.__name__}...")
        result = runner.run(docs_suite())
        if not result.wasSuccessful():
            return 1
        print("=" * 80)

    if args.integration:
        run_integration_tests(args, runner)

    # clean up
    csv_paths = glob.glob("*.csv")
    for csv_path in csv_paths:
        os.remove(csv_path)

    xlsx_paths = glob.glob("*.xlsx")
    for xlsx_path in xlsx_paths:
        os.remove(xlsx_path)

    txt_paths = glob.glob("*.txt")
    for txt_path in txt_paths:
        os.remove(txt_path)

    miro_paths = [
        f"tests{os.sep}conf_test_gamspy",
        f"tests{os.sep}integration{os.sep}conf_test_miro",
        f"tests{os.sep}integration{os.sep}miro_models{os.sep}conf_miro5",
    ]
    for path in miro_paths:
        if os.path.exists(path):
            shutil.rmtree(path)

    misc_paths = [
        "miro.log",
        "gams.gms",
        "HANSEN.GEN",
    ]
    for path in misc_paths:
        if os.path.exists(path):
            os.remove(path)

    shutil.rmtree("tmp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
