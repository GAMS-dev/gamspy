from __future__ import annotations

import atexit
import os
import platform
import signal
import socket
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from typing import TYPE_CHECKING

import gams.transfer as gt

import gamspy as gp
import gamspy._miro as miro
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._backend.backend import backend_factory
from gamspy._extrinsic import ExtrinsicLibrary
from gamspy._miro import MiroJSONEncoder
from gamspy._model import Problem, Sense
from gamspy._options import Options
from gamspy._workspace import Workspace
from gamspy.exceptions import FatalError, GamspyException, ValidationError

if TYPE_CHECKING:
    import io
    from collections.abc import Iterable
    from typing import Any, Literal

    from pandas import DataFrame

    from gamspy import (
        Alias,
        Equation,
        EquationType,
        Model,
        Parameter,
        Set,
        Variable,
    )
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation

LOOPBACK = "127.0.0.1"
IS_MIRO_INIT = os.getenv("MIRO", False)
MIRO_GDX_IN = os.getenv("GAMS_IDC_GDX_INPUT", None)
MIRO_GDX_OUT = os.getenv("GAMS_IDC_GDX_OUTPUT", None)


def open_connection(
    system_directory: str, process_directory: str, license_path: str
) -> tuple[socket.socket, subprocess.Popen]:
    TIMEOUT = 30

    initial_pf_file = os.path.join(process_directory, "gamspy.pf")
    with open(initial_pf_file, "w") as file:
        file.write(
            'incrementalMode="2"\n'
            f'procdir="{process_directory}"\n'
            f'license="{license_path}"\n'
            f'curdir="{os.getcwd()}"\n'
        )

    process = subprocess.Popen(
        [
            os.path.join(system_directory, "gams"),
            "GAMSPY_JOB",
            "pf",
            initial_pf_file,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        start_new_session=platform.system() != "Windows",
    )

    port_info = process.stdout.readline().strip()

    try:
        port = int(port_info.removeprefix("port: "))
    except ValueError as e:
        raise ValidationError(
            f"Error while reading the port! {port_info=}"
        ) from e

    def handler(signum, frame):
        if platform.system() != "Windows":
            os.kill(process.pid, signal.SIGINT)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, handler)

    start = time.time()
    while True:
        if process.poll() is not None:  # pragma: no cover
            raise ValidationError(process.communicate()[0])

        try:
            new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            new_socket.connect((LOOPBACK, port))
            break
        except (ConnectionRefusedError, OSError) as e:
            new_socket.close()
            end = time.time()

            if end - start > TIMEOUT:  # pragma: no cover
                raise FatalError(
                    f"Timeout while establishing the connection with socket. {process.communicate()[0]}"
                ) from e

    return new_socket, process


def get_system_directory(system_directory: str | None) -> str:
    system_directory = os.getenv("GAMSPY_GAMS_SYSDIR", system_directory)

    if system_directory is None:
        system_directory = utils._get_gamspy_base_directory()

    return system_directory


def check_response(response: bytes, job_name: str) -> None:
    GAMS_STATUS = {
        1: "Solver is to be called, the system should never return this number.",
        2: "There was a compilation error.",
        3: "There was an execution error.",
        4: "System limits were reached.",
        5: "There was a file error.",
        6: "There was a parameter error.",
        7: "The solve has failed due to a license error. The license you are using may impose model size limits (demo/community license) or you are using a GAMSPy incompatible professional license. Please contact sales@gams.com to find out about license options.",
        8: "There was a GAMS system error.",
        9: "GAMS could not be started.",
        10: "Out of memory.",
        11: "Out of disk.",
        109: "Could not create process/scratch directory.",
        110: "Too many process/scratch directories.",
        112: "Could not delete the process/scratch directory.",
        113: "Could not write the script gamsnext.",
        114: "Could not write the parameter file.",
        115: "Could not read environment variable.",
        400: "Could not spawn the GAMS language compiler (gamscmex).",
        401: "Current directory (curdir) does not exist.",
        402: "Cannot set current directory (curdir).",
        404: "Blank in system directory (UNIX only).",
        405: "Blank in current directory (UNIX only).",
        406: "Blank in scratch extension (scrext)",
        407: "Unexpected cmexRC.",
        408: "Could not find the process directory (procdir).",
        409: "CMEX library not be found (experimental).",
        410: "Entry point in CMEX library could not be found (experimental).",
        411: "Blank in process directory (UNIX only).",
        412: "Blank in scratch directory (UNIX only).",
        909: "Cannot add path / unknown UNIX environment / cannot set environment variable.",
        1000: "Driver error: incorrect command line parameters for gams.",
        2000: "Driver error: internal error: cannot install interrupt handler.",
        3000: "Driver error: problems getting current directory.",
        4000: "Driver error: internal error: GAMS compile and execute module not found.",
        5000: "Driver error: internal error: cannot load option handling library.",
    }

    value = response[: response.find(b"#")].decode("ascii")
    if not value:
        raise FatalError(
            "Error while getting the return code from GAMS backend. This means that GAMS is in a bad state. Try to backtrack for previous errors."
        )

    return_code = int(value)

    if return_code in GAMS_STATUS:
        try:
            info = GAMS_STATUS[return_code]
        except IndexError:  # pragma: no cover
            info = ""

        raise GamspyException(
            f'{info} Check {job_name + ".lst"} for more information.',
            return_code,
        )


class Container(gt.Container):
    """
    A container is an object that holds all symbols and operates on them.

    Parameters
    ----------
    load_from : str, Container, gt.Container, optional
        Path to the GDX file to be loaded from, by default None
    system_directory : str, optional
        Path to the directory that holds the GAMS installation, by default None
    working_directory : str, optional
        Path to the working directory to store temporary files such .lst, .gms,
        .gdx, .g00 files.
    debugging_level : str, optional
        Decides on keeping the temporary files generate by GAMS, by default
        "keep_on_error"
    options : Options, optional
        Global options for the overall execution

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i")

    """

    def __init__(
        self,
        load_from: str | Container | gt.Container | None = None,
        system_directory: str | None = None,
        working_directory: str | None = None,
        debugging_level: Literal[
            "keep", "keep_on_error", "delete"
        ] = "keep_on_error",
        options: Options | None = None,
        output: io.TextIOWrapper | None = None,
    ):
        self.output = output
        self._gams_string = ""
        if IS_MIRO_INIT:
            atexit.register(self._write_miro_files)

        self._is_socket_open = True

        system_directory = get_system_directory(system_directory)

        self._unsaved_statements: list = []

        super().__init__(system_directory=system_directory)
        self._license_path = utils._get_license_path(self.system_directory)
        self._network_license = self._is_network_license()

        self._debugging_level = debugging_level
        self._workspace = Workspace(debugging_level, working_directory)
        self._working_directory = self._workspace.working_directory
        if len(self._working_directory) > 204:
            raise ValidationError(
                f"Working directory path length cannot be greater than 204! Given working directory path `{self._working_directory}` has length of `{len(self._working_directory)}`"
            )

        self._process_directory = tempfile.mkdtemp(dir=self.working_directory)

        self._job, self._gdx_in, self._gdx_out = self._setup_paths()

        self._temp_container = gt.Container(
            system_directory=self.system_directory
        )

        self._options = validation.validate_global_options(options)

        # needed for miro
        self._miro_input_symbols: list[str] = []
        self._miro_output_symbols: list[str] = []

        self._socket, self._process = open_connection(
            self.system_directory, self._process_directory, self._license_path
        )

        if load_from is not None:
            self._read(load_from)
            if not isinstance(load_from, gt.Container):
                self._unsaved_statements = []
                self._clean_modified_symbols()
                self._add_statement(f"$declareAndLoad {load_from}")

            self._synch_with_gams(gams_to_gamspy=False)

    def __enter__(self):
        pid = os.getpid()
        tid = threading.get_native_id()
        gp._ctx_managers[(pid, tid)] = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pid = os.getpid()
        tid = threading.get_native_id()

        try:
            del gp._ctx_managers[(pid, tid)]
        except KeyError:
            ...

    def __repr__(self) -> str:
        return f"Container(system_directory='{self.system_directory}', working_directory='{self.working_directory}', debugging_level='{self._debugging_level}')"

    def __str__(self):
        if len(self):
            return f"<Container ({hex(id(self))}) with {len(self)} symbols: {self.data.keys()}>"

        return f"<Empty Container ({hex(id(self))})>"

    def __del__(self):
        try:
            self._stop_socket()
        except (Exception, ConnectionResetError):  # pragma: no cover
            ...

    @property
    def working_directory(self) -> str:
        """
        Working directory path.

        Returns
        -------
        str
        """
        return self._working_directory

    @property
    def in_miro(self) -> bool:
        """
        When running a GAMSPy job from GAMS MIRO, you may not want to
        perform certain expensive operations, such as loading MIRO input
        data from an Excel workbook, as this data comes from MIRO. In that
        case, one can conditionally load the data by using the ``in_miro``
        attribute of `Container`.

        Returns
        -------
        bool
        """
        return MIRO_GDX_IN is not None

    def _is_network_license(self) -> bool:
        with open(self._license_path, encoding="utf-8") as file:
            lines = file.readlines()

        return bool("+" in lines[0] and lines[4][47] == "N")

    def _stop_socket(self):
        if hasattr(self, "_socket") and self._is_socket_open:
            self._socket.sendall(b"stop")
            self._socket.close()
            self._is_socket_open = False

            if not self._process.stdout.closed:
                self._process.stdout.close()

            while self._process.poll() is None:
                ...

    def _read_output(self, output: io.TextIOWrapper | None) -> None:
        if output is not None:
            while True:
                data = self._process.stdout.readline()
                output.write(data)
                output.flush()
                if data.startswith("--- Job ") and "elapsed" in data:
                    break

    def _interrupt(self) -> None:
        if platform.system() == "Windows":
            os.kill(self._process.pid, signal.CTRL_C_EVENT)
        else:
            os.kill(self._process.pid, signal.SIGINT)

    def _send_job(
        self,
        job_name: str,
        pf_file: str,
        output: io.TextIOWrapper | None = None,
    ):
        try:
            # Send pf file
            self._socket.sendall(pf_file.encode("utf-8"))

            # Read output
            self._read_output(output)

            # Receive response
            response = self._socket.recv(256)
            check_response(response, job_name)
        except ConnectionError as e:
            raise FatalError(
                f"There was an error while communicating with GAMS server: {e}",
            ) from e

    def _write_miro_files(self):
        # create conf_<model>/<model>_io.json
        try:
            encoder = MiroJSONEncoder(self)
            encoder.write_json()
        except Exception:
            self._stop_socket()
            traceback.print_exc()
            os._exit(1)

    def _add_statement(self, statement) -> None:
        self._unsaved_statements.append(statement)

    def _cast_symbols(self, symbol_names: list[str] | None = None) -> None:
        """Casts GTP symbols to GAMSPy symbols"""
        symbol_names = symbol_names if symbol_names else list(self.data.keys())

        for symbol_name in symbol_names:
            gtp_symbol = self.data[symbol_name]
            new_domain = [
                (
                    self.data[member.name]
                    if not isinstance(member, str)
                    else member
                )
                for member in gtp_symbol.domain
            ]

            del self.data[symbol_name]

            if isinstance(gtp_symbol, gt.Alias):
                alias_with = self.data[gtp_symbol.alias_with.name]
                _ = gp.Alias._constructor_bypass(
                    self, gtp_symbol._name, alias_with
                )
            elif isinstance(gtp_symbol, gt.UniverseAlias):
                _ = gp.UniverseAlias._constructor_bypass(
                    self,
                    gtp_symbol._name,
                )
            elif isinstance(gtp_symbol, gt.Set):
                _ = gp.Set._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    new_domain,
                    gtp_symbol._is_singleton,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )
            elif isinstance(gtp_symbol, gt.Parameter):
                _ = gp.Parameter._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    new_domain,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )
            elif isinstance(gtp_symbol, gt.Variable):
                _ = gp.Variable._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    gtp_symbol._type,
                    new_domain,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )
            elif isinstance(gtp_symbol, gt.Equation):
                symbol_type = gtp_symbol.type
                if gtp_symbol.type in ["eq", "leq", "geq"]:
                    symbol_type = "regular"

                _ = gp.Equation._constructor_bypass(
                    self,
                    gtp_symbol._name,
                    symbol_type,
                    new_domain,
                    gtp_symbol._records,
                    gtp_symbol._description,
                )

    def _delete_autogenerated_symbols(self) -> None:
        """
        Removes autogenerated model attributes, objective variable and equation from
        the container
        """
        autogenerated_symbol_names = self._get_autogenerated_symbol_names()

        for name in autogenerated_symbol_names:
            if name in self.data:
                del self.data[name]

    def _get_symbol_names_to_load(
        self, load_from: str, names: list[str] | None
    ) -> list[str]:
        if names is None:
            names = utils._get_symbol_names_from_gdx(
                self.system_directory, load_from
            )

        return names

    def _setup_paths(self) -> tuple[str, str, str]:
        suffix = "_" + str(uuid.uuid4())
        job = os.path.join(self.working_directory, suffix)
        gdx_in = os.path.join(self.working_directory, f"{suffix}in.gdx")
        gdx_out = os.path.join(self.working_directory, f"{suffix}out.gdx")

        return job, gdx_in, gdx_out

    def _get_autogenerated_symbol_names(self) -> list[str]:
        names = []
        for name in self.data:
            if name.startswith(gp.Model._generate_prefix):
                names.append(name)

        return names

    def _get_modified_symbols(self) -> list[str]:
        modified_names = []

        for name, symbol in self:
            if symbol.modified:
                if (
                    isinstance(symbol, gp.Alias)
                    and symbol.alias_with.name not in modified_names  # type: ignore
                ):
                    modified_names.append(symbol.alias_with.name)

                modified_names.append(name)

        return modified_names

    def _clean_modified_symbols(self) -> None:
        for symbol in self.data.values():
            if isinstance(symbol, gp.Alias):
                symbol.alias_with.modified = False

            symbol.modified = False

    def _synch_with_gams(
        self,
        relaxed_domain_mapping: bool = False,
        gams_to_gamspy: bool = False,
    ) -> DataFrame | None:
        runner = backend_factory(self, self._options, output=self.output)
        summary = runner.run(relaxed_domain_mapping, gams_to_gamspy)

        if self._options and self._options.seed is not None:
            # Required for correct seeding. Seed can only be set in the first run.
            self._options.seed = None

        if IS_MIRO_INIT:
            miro._write_default_gdx_miro(self)

        return summary

    def _generate_gams_string(
        self,
        gdx_in: str,
        modified_names: list[str],
    ) -> str:
        LOADABLE = (gp.Set, gp.Parameter, gp.Variable, gp.Equation)
        MIRO_INPUT_TYPES = (gt.Set, gt.Parameter)

        strings = ["$onMultiR", "$onUNDF"]
        for statement in self._unsaved_statements:
            if isinstance(statement, str):
                strings.append(statement)
            else:
                strings.append(statement.getDeclaration())

        if modified_names:
            loadables = []
            for name in modified_names:
                symbol = self[name]
                if (
                    isinstance(symbol, LOADABLE)
                    and not name.startswith(gp.Model._generate_prefix)
                    and symbol.synchronize
                ):
                    loadables.append(symbol)

            if loadables:
                strings.append(f"$gdxIn {gdx_in}")
                for loadable in loadables:
                    if (
                        isinstance(loadable, MIRO_INPUT_TYPES)
                        and loadable._is_miro_input
                        and not IS_MIRO_INIT
                        and MIRO_GDX_IN
                    ):
                        miro_load = miro.get_load_input_str(
                            loadable.name, gdx_in
                        )
                        strings.append(miro_load)
                    else:
                        strings.append(f"$loadDC {loadable.name}")

                strings.append("$gdxIn")

        strings.append("$offUNDF")

        if not IS_MIRO_INIT and MIRO_GDX_OUT:
            strings.append(miro.get_unload_output_str(self))

        gams_string = "\n".join(strings)

        if self._debugging_level == "keep":
            self._gams_string += gams_string + "\n"

        return gams_string

    def _load_records_from_gdx(
        self,
        load_from: str,
        symbol_names: list[str] | None = None,
        user_invoked: bool = False,
    ):
        symbol_names = self._get_symbol_names_to_load(load_from, symbol_names)

        self._temp_container.read(load_from, symbol_names)

        for name in symbol_names:
            if name in self.data:
                updated_records = self._temp_container[name].records

                if user_invoked:
                    self[name].records = updated_records
                else:
                    self[name]._records = updated_records

                if updated_records is not None:
                    self[name].domain_labels = self[name].domain_names
            else:
                self._read(load_from, [name])

            if user_invoked:
                self[name].modified = True

        self._temp_container.data = {}

        if user_invoked:
            self._synch_with_gams()

    def _read(
        self,
        load_from: str | Container | gt.Container,
        symbol_names: list[str] | None = None,
        load_records: bool = True,
        mode: str | None = None,
        encoding: str | None = None,
    ) -> None:
        super().read(load_from, symbol_names, load_records, mode, encoding)
        self._cast_symbols(symbol_names)

    def read(
        self,
        load_from: str | Container | gt.Container,
        symbol_names: list[str] | None = None,
        load_records: bool = True,
        mode: str | None = None,
        encoding: str | None = None,
    ) -> None:
        """
        Reads specified symbols from the GDX file. If symbol_names are
        not provided, it reads all symbols from the GDX file.

        Parameters
        ----------
        load_from : str, Container, gt.Container
        symbol_names : List[str], optional
        load_records : bool
        mode : str, optional
        encoding : str, optional

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1', 'i2'])
        >>> m.write("test.gdx")
        >>> new_container = gp.Container()
        >>> new_container.read("test.gdx")
        >>> new_container.data.keys() == m.data.keys()
        True

        """
        self._read(load_from, symbol_names, load_records, mode, encoding)
        self._synch_with_gams()

    def write(
        self,
        write_to: str,
        symbol_names: list[str] | None = None,
        compress: bool = False,
        mode: str | None = None,
        eps_to_zero: bool = True,
    ) -> None:
        """
        Writes specified symbols to the GDX file. If symbol_names are
        not provided, it writes all symbols to the GDX file.

        Parameters
        ----------
        write_to : str
        symbol_names : List[str], optional
        compress : bool
        mode : str, optional
        eps_to_zero : bool

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1', 'i2'])
        >>> m.write("test.gdx")

        """
        super().write(
            write_to,
            symbol_names,
            compress,
            mode=mode,
            eps_to_zero=eps_to_zero,
        )

    def generateGamsString(self, show_raw: bool = False) -> str:
        """
        Generates the GAMS code

        Parameters
        ----------
        show_raw : bool, optional
            Shows the raw model without data and other necessary
            GAMS statements, by default False.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container(debugging_level="keep")
        >>> i = gp.Set(m, name="i")
        >>> gams_code = m.generateGamsString()

        """
        if self._debugging_level != "keep":
            raise ValidationError(
                "`debug_level` argument of the container must be set to 'keep' to use this function."
            )

        if not show_raw:
            return self._gams_string

        return utils._filter_gams_string(self._gams_string)

    def loadRecordsFromGdx(
        self,
        load_from: str,
        symbol_names: list[str] | None = None,
    ) -> None:
        """
        Loads data of the given symbols from a GDX file. If no
        symbol names are given, data of all symbols are loaded.

        Parameters
        ----------
        load_from : str
            Path to the GDX file
        symbols : List[str], optional
            Symbols whose data will be load from GDX, by default None

        Examples
        --------
        >>> from gamspy import Container, Set
        >>> m = Container()
        >>> i = Set(m, "i", records=["i1", "i2"])
        >>> m.write("test.gdx")
        >>> m2 = Container()
        >>> m2.loadRecordsFromGdx("test.gdx")
        >>> print(i.records.equals(m2["i"].records))
        True

        """
        self._load_records_from_gdx(load_from, symbol_names, user_invoked=True)

    def addGamsCode(self, gams_code: str) -> None:
        """
        Adds an arbitrary GAMS code to the generate .gms file.
        Using addGAMSCode might result in a license error if no GAMSpy++ license is used.

        Parameters
        ----------
        gams_code : str
            GAMS code that you want to insert.

        Examples
        --------
        >>> from gamspy import Container
        >>> m = Container()
        >>> m.addGamsCode("scalar piHalf / [pi/2] /;")
        >>> m["piHalf"].toValue()
        np.float64(1.5707963267948966)

        """
        self._add_statement(gams_code)
        self._synch_with_gams(relaxed_domain_mapping=True, gams_to_gamspy=True)

        for _, symbol in self:
            symbol.modified = False
        self._unsaved_statements = []

    def close(self) -> None:
        """
        Stops the socket and releases resources. The container should not be used afterwards
        to communicate with the GAMS execution engine, e.g. creating new symbols, changing data,
        solves, etc. The container data (Container.data) is still available for read operations.
        """
        self._stop_socket()

    def addAlias(
        self,
        name: str | None = None,
        alias_with: Set | Alias = None,  # type: ignore
    ) -> Alias:
        """
        Creates a new Alias and adds it to the container

        Parameters
        ----------
        name : str, optional
            Name of the alias.
        alias_with : Set | Alias
            Alias set object.

        Returns
        -------
        Alias

        Raises
        ------
        TypeError
            In case the alias_with is different than a Set or an Alias
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")
        >>> a = m.addAlias("a", i)

        """
        return gp.Alias(self, name, alias_with)

    def addSet(
        self,
        name: str | None = None,
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
    ) -> Set:
        """
        Creates a Set and adds it to the container

        Parameters
        ----------
        name : str, optional
            Name of the set. Name is autogenerated by default.
        domain : list[Set | Alias | str] | Set | Alias | str, optional
            Domain of the set.
        is_singleton : bool, optional
            Whether the set is a singleton set. Singleton sets cannot contain more than one element.
        records : pd.DataFrame | np.ndarray | list, optional
            Records of the set.
        domain_forwarding : bool, optional
            Whether the set forwards the domain.
        description : str, optional
            Description of the set.
        uels_on_axes : bool
            Assume that symbol domain information is contained in the axes of the given records.
        is_miro_input : bool
            Whether the symbol is a GAMS MIRO input symbol. See: https://gams.com/miro/tutorial.html
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Set

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            When there is symbol with same name in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addSet("i")

        """
        return gp.Set(
            self,
            name,
            domain,
            is_singleton,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_input=is_miro_input,
            is_miro_output=is_miro_output,
        )

    def addParameter(
        self,
        name: str | None = None,
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ) -> Parameter:
        """
        Creates a Parameter and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the parameter. Name is autogenerated by default.
        domain : list[Set | Alias | str] | Set | Alias | str, optional
            Domain of the parameter.
        records : int | float | pd.DataFrame | np.ndarray | list, optional
            Records of the parameter.
        domain_forwarding : bool, optional
            Whether the parameter forwards the domain.
        description : str, optional
            Description of the parameter.
        uels_on_axes : bool
            Assume that symbol domain information is contained in the axes of the given records.
        is_miro_input : bool
            Whether the symbol is a GAMS MIRO input symbol. See: https://gams.com/miro/tutorial.html
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html
        is_miro_table : bool
            Whether the symbol is a GAMS MIRO table symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Parameter

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = m.addParameter("a")

        """
        return gp.Parameter(
            self,
            name,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_input=is_miro_input,
            is_miro_output=is_miro_output,
            is_miro_table=is_miro_table,
        )

    def addVariable(
        self,
        name: str | None = None,
        type: str = "free",
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
    ) -> Variable:
        """
        Creates a Variable and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the variable. Name is autogenerated by default.
        type : str, optional
            Type of the variable. "free" by default.
        domain : list[Set | Alias | str] | Set | Alias | str, optional
            Domain of the variable.
        records : Any, optional
            Records of the variable.
        domain_forwarding : bool, optional
            Whether the variable forwards the domain.
        description : str, optional
            Description of the variable.
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Variable

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> v = m.addVariable("v")

        """
        return gp.Variable(
            self,
            name,
            type,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_output=is_miro_output,
        )

    def addEquation(
        self,
        name: str | None = None,
        type: str | EquationType = "regular",
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
        definition_domain: list | None = None,
    ) -> Equation:
        """
        Creates an Equation and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the equation. Name is autogenerated by default.
        type : str
            Type of the equation. "regular" by default.
        domain : list[Set | Alias | str] | Set | Alias | str, optional
            Domain of the variable.
        definition: Expression, optional
            Definition of the equation.
        records : Any, optional
            Records of the equation.
        domain_forwarding : bool, optional
            Whether the equation forwards the domain.
        description : str, optional
            Description of the equation.
        uels_on_axes: bool
            Assume that symbol domain information is contained in the axes of the given records.
        definition_domain: list, optional
            Definiton domain of the equation.
        is_miro_output : bool
            Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

        Returns
        -------
        Equation

        Raises
        ------
        err
            In case arguments are not valid
        ValueError
            If there is symbol with same name but different type in the
            Container

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = m.addEquation("i")

        """
        return gp.Equation(
            self,
            name,
            type,
            domain,
            definition,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
            is_miro_output,
            definition_domain,
        )

    def addModel(
        self,
        name: str | None = None,
        problem: Problem | str = Problem.MIP,
        equations: Iterable[Equation] = [],
        sense: Sense | str = Sense.FEASIBILITY,
        objective: Variable | Expression | None = None,
        matches: dict[Equation, Variable] | None = None,
        limited_variables: Iterable[Variable] | None = None,
        external_module: str | None = None,
    ) -> Model:
        """
        Creates a Model and adds it to the Container

        Parameters
        ----------
        name : str, optional
            Name of the model. Name is autogenerated by default.
        equations : Iterable[Equation]
            Iterable of Equation objects.
        problem : Problem or str, optional
            'LP', 'NLP', 'QCP', 'DNLP', 'MIP', 'RMIP', 'MINLP', 'RMINLP', 'MIQCP', 'RMIQCP', 'MCP', 'CNS', 'MPEC', 'RMPEC', 'EMP', or 'MPSGE',
            by default Problem.LP.
        sense : Sense, optional
            "MIN", "MAX", or "FEASIBILITY".
        objective : Variable | Expression, optional
            Objective variable to minimize or maximize or objective itself.
        matches : dict[Equation, Variable]
            Equation - Variable matches for MCP models.
        limited_variables : Iterable, optional
            Allows limiting the domain of variables used in a model.
        external_module: str, optional
            The name of the external module in which the external equations are implemented

        Returns
        -------
        Model

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = gp.Equation(m, "e")
        >>> model = m.addModel("my_model", "LP", [e])

        """
        return gp.Model(
            self,
            name,
            problem,
            equations,
            sense,
            objective,
            matches,
            limited_variables,
            external_module=external_module,
        )

    def copy(self, working_directory: str) -> Container:
        """
        Creates a copy of the Container. Should not be invoked after
        creating the model.

        Parameters
        ----------
        working_directory : str, optional
            Working directory of the new Container, by default None

        Returns
        -------
        Container

        Raises
        ------
        ValidationError

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i")
        >>> new_cont = m.copy(working_directory="test")
        >>> new_cont.data.keys() == m.data.keys()
        True

        """
        os.makedirs(working_directory, exist_ok=True)
        m = Container(working_directory=working_directory)
        if m.working_directory == self.working_directory:
            raise ValidationError(
                "Copy of a container cannot have the same working directory"
                " with the original container."
            )

        self.write(m._job + "in.gdx")
        m._read(m._job + "in.gdx")

        # if already defined equations exist, add them to .gms file
        for equation in self.getEquations():
            if equation._definition is not None:
                m._add_statement(equation._definition)
                m[equation.name]._definition = equation._definition
                m._synch_with_gams()

        return m

    def getEquations(self) -> list[Equation]:
        """
        Returns all equation symbols in the Container.

        Returns
        -------
        list[Equation]

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> eq1 = gp.Equation(m, name="eq1")
        >>> eq2 = gp.Equation(m, name="eq2")
        >>> equation_objects = m.getEquations()

        """
        equations = [
            equation
            for equation in self.listEquations()
            if not equation.startswith(gp.Model._generate_prefix)
        ]
        return self.getSymbols(equations)

    def importExtrinsicLibrary(
        self, lib_path: str, functions: dict[str, str]
    ) -> ExtrinsicLibrary:
        """
        Imports an extrinsic library to the GAMS environment.

        Parameters
        ----------
        lib_path : str
            Path to the .so, .dylib or .dll file that contains the extrinsic library
        functions : dict[str, str]
            Names of the functions as a dictionary. Key is the desired function name in GAMSPy
            and value is the function name in the extrinsic library.

        Returns
        -------
        ExtrinsicLibrary

        Raises
        ------
        FileNotFoundError
            In case the extrinsic library does not exist in the given path.
        """
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"`{lib_path}` is not a valid path.")

        external_lib = ExtrinsicLibrary(lib_path, functions)
        self._add_statement(external_lib)

        return external_lib

    def gamsJobName(self) -> str | None:
        """
        Returns the name of the latest GAMS job that was executed

        Returns
        -------
        str | None

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["seattle", "san-diego"], description="canning plants")
        >>> gams_file_name = f"{m.gamsJobName()}.gms"

        """
        return self._job

    def gdxInputPath(self) -> str:
        """
        Path to the input GDX file

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["seattle", "san-diego"], description="canning plants")
        >>> gdx_path = m.gdxInputPath()

        """
        return self._gdx_in

    def gdxOutputPath(self) -> str:
        """
        Path to the output GDX file

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"], description="canning plants")
        >>> ii = gp.Set(m, name="ii", domain=i, description="seattle plant")
        >>> ii['seattle'] = True
        >>> gdx_path = m.gdxOutputPath()

        """
        return self._gdx_out
