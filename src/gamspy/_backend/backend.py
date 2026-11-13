from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TextIO, no_type_check

import pandas as pd

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

    from gamspy import Container, Model, Options
    from gamspy._backend.engine import EngineClient, GAMSEngine
    from gamspy._backend.local import Local
    from gamspy._backend.neos import NeosClient, NEOSServer

HEADER = [
    "Solver Status",
    "Model Status",
    "Objective",
    "Num of Equations",
    "Num of Variables",
    "Model Type",
    "Solver",
    "Solver Time",
]


@no_type_check
def backend_factory(
    container: Container,
    options: Options | None = None,
    solver: str | None = None,
    solver_options: dict | Path | None = None,
    output: TextIO | None = None,
    backend: Literal["local", "engine", "neos"] = "local",
    client: EngineClient | NeosClient | None = None,
    model: Model | None = None,
) -> Local | GAMSEngine | NEOSServer:  # pragma: no cover
    if backend == "neos":
        from gamspy._backend.neos import NeosClient, NEOSServer

        if client is None or not isinstance(client, NeosClient):
            raise ValidationError(
                "`NeosClient` must be provided to solve on NEOS Server."
            )

        return NEOSServer(
            container,
            options,
            solver,
            solver_options,
            output,
            model,
            client,
        )
    elif backend == "engine":
        from gamspy._backend.engine import EngineClient, GAMSEngine

        if client is None or not isinstance(client, EngineClient):
            raise ValidationError(
                "`engine_client` must be provided to solve on GAMS Engine"
            )

        return GAMSEngine(
            container,
            options,
            solver,
            solver_options,
            output,
            model,
            client,
        )
    elif backend == "local":
        from gamspy._backend.local import Local

        return Local(
            container,
            options,
            solver,
            solver_options,
            output,
            model,
        )

    raise ValidationError(
        f"`{backend}` is not a valid backend. Possible backends:"
        " local, engine, and neos"
    )


class Backend(ABC):
    def __init__(
        self,
        backend_type: str,
        container: Container,
        options: Options,
        solver: str | None,
        solver_options: dict | Path | None,
        output: TextIO | None,
    ) -> None:
        self.backend_type = backend_type
        self.container = container
        self.options = options
        self.solver = solver
        self.solver_options = solver_options
        self.output = output

        self.job_name = self.get_job_name()
        self.gms_file = self.job_name + ".gms"
        self.lst_file = self.job_name + ".lst"
        self.pf_file = self.job_name + ".pf"
        self.restart_file = self.job_name + ".g00"
        self.symbol_names: list[str] = []

    @abstractmethod
    def is_async(self) -> bool: ...

    @abstractmethod
    def run(self) -> pd.DataFrame | None: ...

    @abstractmethod
    def create_handle(self) -> SolveHandle:
        """Handle of the solve this backend just submitted asynchronously."""

    def get_job_name(self) -> str:
        job_name = self.container._job

        if self.container._debugging_level == "keep":
            job_name = os.path.join(
                self.container.working_directory,
                "_" + utils._get_unique_name(),
            )
            self.container._job = job_name
            self.container._gdx_in = f"{job_name}in.gdx"
            self.container._gdx_out = f"{job_name}out.gdx"

        return job_name

    def preprocess(self) -> str:
        symbol_names = self.container._symbols_to_unload()
        self.symbol_names = symbol_names

        if len(symbol_names) != 0:
            self.container._write(
                self.container._gdx_in, symbol_names, eps_to_zero=False
            )

        gdx_in = self.container._gdx_in
        if self.backend_type == "engine":
            gdx_in = os.path.basename(gdx_in)
        elif self.backend_type == "neos":
            gdx_in = "in.gdx"
        gams_string = self.container._generate_gams_string(gdx_in, symbol_names)

        return gams_string

    def _mark_unloaded_symbols_dirty(self) -> None:
        for name in self.symbol_names:
            symbol = self.container._data.get(name)
            if symbol is not None:
                symbol._should_unload_to_gams = True

    @staticmethod
    def _copy_log(log_path: str, options: Options) -> None:
        if options.log_file is None:
            return

        os.makedirs(os.path.dirname(os.path.abspath(options.log_file)), exist_ok=True)
        mode = "a" if options.append_to_log_file else "w"
        with (
            open(log_path, encoding="utf-8") as source,
            open(options.log_file, mode, encoding="utf-8") as target,
        ):
            target.write(source.read())

    def parse_listings(self, model: Model) -> None:
        listing_file = (
            self.options.listing_file
            if self.options.listing_file
            else self.job_name + ".lst"
        )
        if self.options.equation_listing_limit:
            utils._parse_generated_equations(model, listing_file)

        if self.options.variable_listing_limit:
            utils._parse_generated_variables(model, listing_file)

    def prepare_summary(self, model: Model) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                [
                    model.solve_status.name,
                    model.status.name,
                    model.objective_value,
                    model.num_equations,
                    model.num_variables,
                    model.used_model_type,
                    self.solver.upper(),  # ty: ignore[missing-union-attribute]
                    model.total_solver_time,
                ]
            ],
            columns=HEADER,
        )
        return df


class SolveHandle(ABC):
    """
    Handle of an asynchronous solve, returned by
    :meth:`Model.solveAsync <gamspy.Model.solveAsync>`.

    Each asynchronous solve submits a separate model instance and returns its
    own handle, so several solves of the same model can be in flight at once.
    The solution of an instance is loaded into the model with :meth:`wait`.
    Read the records you need before collecting the next handle, since every
    collection overwrites the symbol records and model attributes.

    Every backend returns its own subclass but they all implement :meth:`isReady` and :meth:`wait`.

    This class is not meant to be instantiated directly.
    """

    # Name of the backend the solve was submitted to.
    backend: str

    def __init__(self, model: Model) -> None:
        self.model = model
        self._is_collected = False

    @property
    @abstractmethod
    def _job_id(self) -> object:
        """Identity of the submitted job on its backend."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model='{self.model.name}',"
            f" job={self._job_id!r}, collected={self._is_collected})"
        )

    def _validate(self) -> None:
        if self.model.container._in_loop:
            raise ValidationError(
                "`SolveHandle` methods cannot be used inside control flow blocks (Loop, If, While etc.)."
            )

        if self._is_collected:
            raise ValidationError(
                f"The solution of this asynchronous solve of model `{self.model.name}` has already been collected."
            )

    @staticmethod
    def _validate_timeout(timeout: float | None) -> None:
        if timeout is not None and timeout < 0:
            raise ValidationError("`timeout` must be nonnegative.")

    @abstractmethod
    def isReady(self) -> bool:
        """
        Returns True if the solver is done.

        Returns
        -------
        bool

        Raises
        ------
        ValidationError
            In case the solution has already been collected.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition=v >= 5)
        >>> model = gp.Model(m, "my_model", equations=[e], problem="LP", sense="MIN", objective=v)
        >>> handle = model.solveAsync()
        >>> import time
        >>> while not handle.isReady():
        ...     time.sleep(1)  # do other work
        >>> summary = handle.wait()

        """

    @abstractmethod
    def wait(
        self,
        timeout: float | None = None,
        output: TextIO | None = None,
        options: Options | None = None,
    ) -> pd.DataFrame:
        """
        Waits for the asynchronous solve to finish and loads its solution.

        Blocks until the solver is done, loads the solution and the model
        attributes into the model and returns the solve summary.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds. If the solve does not finish in
            time, a TimeoutError is raised and the solve keeps running in the
            background; `wait` can be called again later. Waits indefinitely
            by default. A timeout of 0 checks the status without waiting.
        output : TextIO, optional
            Output redirection target. Set this to sys.stdout to see the output in your terminal.
        options : Options, optional
            GAMSPy options of the collection. The solver log is written to
            ``log_file``, appended if ``append_to_log_file`` is set.

        Returns
        -------
        DataFrame
            Summary of the solve

        Raises
        ------
        ValidationError
            In case the solution has already been collected.
        TimeoutError
            In case the solve did not finish within `timeout` seconds.
        GamspyException
            In case the solution of the finished solve cannot be retrieved.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = gp.Variable(m, "v")
        >>> e = gp.Equation(m, "e", definition=v >= 5)
        >>> model = gp.Model(m, "my_model", equations=[e], problem="LP", sense="MIN", objective=v)
        >>> handle = model.solveAsync()
        >>> summary = handle.wait()
        >>> model.objective_value
        5.0

        """
