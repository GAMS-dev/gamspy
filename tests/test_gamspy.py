import argparse
import glob
import os
import unittest

from integration.test_cmd_script import cmd_suite
from integration.test_model_instance import model_instance_suite
from integration.test_models import gams_models_suite
from integration.test_solve import solve_suite
from unit.test_alias import alias_suite
from unit.test_condition import condition_suite
from unit.test_container import container_suite
from unit.test_domain import domain_suite
from unit.test_equation import equation_suite
from unit.test_magics import magics_suite
from unit.test_math import math_suite
from unit.test_model import model_suite
from unit.test_operation import operation_suite
from unit.test_parameter import parameter_suite
from unit.test_set import set_suite
from unit.test_utils import utils_suite
from unit.test_variable import variable_suite


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    runner = unittest.TextTestRunner()

    unittest_suites = [
        container_suite,
        math_suite,
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
    ]

    print(f"Running unittests\n{'='*80}")
    for suite in unittest_suites:
        print("=" * 80)
        print(f"\nRunning {suite.__name__}...")
        result = runner.run(suite())
        if not result.wasSuccessful():
            return 1
        print("=" * 80)

    if args.integration:
        integration_suites = [
            solve_suite,
            model_instance_suite,
            gams_models_suite,
            cmd_suite,
        ]

        print(f"Running integration tests\n{'='*80}")
        for suite in integration_suites:
            print("=" * 80)
            print(f"\nRunning {suite.__name__}...")
            result = runner.run(suite())
            if not result.wasSuccessful():
                return 1
            print("=" * 80)

    # clean up
    csv_paths = glob.glob("*.csv")
    for csv_path in csv_paths:
        os.remove(csv_path)

    xlsx_paths = glob.glob("*.xlsx")
    for xlsx_path in xlsx_paths:
        os.remove(xlsx_path)

    txt_paths = glob.glob("*.txt")
    for txt_path in txt_paths:
        if txt_path.endswith("dev_requirements.txt"):
            continue
        os.remove(txt_path)

    gdx_paths = glob.glob("*.gdx")
    for gdx_path in gdx_paths:
        os.remove(gdx_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
