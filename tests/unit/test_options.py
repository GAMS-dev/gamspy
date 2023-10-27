import unittest

from pydantic import ValidationError

from gamspy import Options


class OptionsSuite(unittest.TestCase):
    def test_options(self):
        with self.assertRaises(ValidationError):
            _ = Options(action=5)

        options = Options(action="restart_after_solve")
        self.assertEqual(options.action, "R")

        with self.assertRaises(ValidationError):
            _ = Options(append_output=5)

        options = Options(append_output=True)
        self.assertEqual(options.append_output, 1)

        with self.assertRaises(ValidationError):
            _ = Options(report_async_solve=5)

        options = Options(report_async_solve=True)
        self.assertEqual(options.report_async_solve, 1)

        with self.assertRaises(ValidationError):
            _ = Options(hold_fixed_variables=5)

        options = Options(hold_fixed_variables=True)
        self.assertEqual(options.hold_fixed_variables, 1)

        with self.assertRaises(ValidationError):
            _ = Options(hold_fixed_variables_async=5)

        options = Options(hold_fixed_variables_async=True)
        self.assertEqual(options.hold_fixed_variables_async, 1)

        with self.assertRaises(ValidationError):
            _ = Options(report_solution=5)

        options = Options(report_solution=True)
        self.assertEqual(options.report_solution, 1)

        with self.assertRaises(ValidationError):
            _ = Options(multi_solve_strategy=5)

        options = Options(multi_solve_strategy="replace")
        self.assertEqual(options.multi_solve_strategy, 0)

        with self.assertRaises(ValidationError):
            _ = Options(step_summary=5)

        options = Options(step_summary=True)
        self.assertEqual(options.step_summary, 1)

        with self.assertRaises(ValidationError):
            _ = Options(suppress_compiler_listing=5)

        options = Options(suppress_compiler_listing=True)
        self.assertEqual(options.suppress_compiler_listing, 1)

        with self.assertRaises(ValidationError):
            _ = Options(report_solver_status=5)

        options = Options(report_solver_status=True)
        self.assertEqual(options.report_solver_status, 1)

        with self.assertRaises(ValidationError):
            _ = Options(report_underflow=5)

        options = Options(report_underflow=True)
        self.assertEqual(options.report_underflow, 1)


def options_suite():
    suite = unittest.TestSuite()
    tests = [
        OptionsSuite(name)
        for name in dir(OptionsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(options_suite())
