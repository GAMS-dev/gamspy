# Architecture

## Non-obvious modules (`src/gamspy`)

- `_container.py`: `Container`. It owns all symbols, the queue of unsaved GAMS statements, the working directory and the execution engine.
- `_symbols/implicits/`: symbol suffixes, attributes and indexed views (for example `x.l[i]`), as opposed to the top-level symbols in `_symbols/`.
- `_backend/` vs `_execution/`: `_backend/` chooses *where* a job runs (`Local`, `Engine`, `Neos` through `backend_factory`). `_execution/` is *how* the local backend talks to the GAMS process (`SocketEngine`).
- `_frozen_model.py`: Allows generating the model instance once from the algebra and solving it with modified data (`Model.freeze`).
- `_options.py` vs `_config.py`: `_options.py` holds per-solve Pydantic options (`Options`, `FreezeOptions`, `ConvertOptions`). `_config.py` holds package-wide settings (`set_options`/`get_option`).
- `_validation.py`: checks on user input.
- `src/gamspy_cli/`: the `gamspy` CLI (Typer), with one module per subcommand.

## How a statement reaches GAMS

1. An assignment or declaration (for example `x[i] = ...`) builds an `Expression`, and `Container._add_statement` queues it.
2. `Container._synch_with_gams()` (or `Model.solve()`) calls `backend_factory(...)`, whose `run()` method:
   - `preprocess()`: writes the changed symbols to the input GDX and builds the GAMS string with `Container._generate_gams_string`.
   - `execute_gams()`: sends the string to `container._execution_engine`, then clears the queued statements.
   - `postprocess()`: loads the results back and, after a solve, builds the summary DataFrame.
3. Inside control-flow blocks (`For`, `If`, `Loop`, `While` in `_control_flow.py`), `container._in_loop` defers synchronization, so the statements are emitted as GAMS control flow.

## Debugging generated code

- To check the GAMS representation of an expression, call `expression.gamsRepr()`.
- To check all executed GAMS code, create the container with `debugging_level="keep"` and call `container.generateGamsString()`.
