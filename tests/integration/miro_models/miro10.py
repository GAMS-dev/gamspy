import argparse

parser = argparse.ArgumentParser(description="Delete files safely.")

# Get the kwarg option
parser.add_argument("--my-kwarg")

# Get the positional args
parser.add_argument("files", nargs="*", help="Files to delete")

args = parser.parse_args()

assert args.my_kwarg == "bla"
assert args.files == ["a", "b", "c", "d"]
