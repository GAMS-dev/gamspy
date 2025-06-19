from __future__ import annotations

import base64
import logging
import os
import shutil
import ssl
import xmlrpc.client
import zipfile
from typing import TYPE_CHECKING

import certifi

import gamspy._backend.backend as backend
import gamspy.utils as utils
from gamspy._options import Options
from gamspy.exceptions import (
    GamspyException,
    NeosClientException,
    ValidationError,
)

logger = logging.getLogger("NEOS")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s - %(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if TYPE_CHECKING:
    import io

    from gamspy import Container, Model
    from gamspy._symbols.symbol import Symbol


class NeosClient:
    def __init__(
        self,
        email: str,
        server: str = "https://neos-server.org:3333",
        username: str | None = None,
        password: str | None = None,
        priority: str = "long",
        is_blocking: bool = True,
    ) -> None:
        """
        Python client for NEOS Server. Implements the functions listed in:
        https://neos-server.org/neos/xml-rpc.html

        Parameters
        ----------
        email : str
        server : _type_, optional
            Server ip and port, by default "https://neos-server.org:3333"
        username : Optional[str], optional
            Username, by default None
        password : Optional[str], optional
            Password, by default None
        priority : str, optional
            Priority of the job, by default "long"
        is_blocking : bool, optional
            Decides on sync or async execution, by default True
        """
        self.email = email
        self.server = server
        self.username = username
        self.password = password
        self.priority = priority
        self.is_blocking = is_blocking
        context = ssl.create_default_context(cafile=certifi.where())
        self.neos = xmlrpc.client.ServerProxy(server, context=context)
        self.jobs: list[tuple] = []

    def is_alive(self) -> bool:
        """
        Checks if NEOS Server is alive

        Returns
        -------
        bool
        """
        response: str = self.neos.ping()
        return bool(response.startswith("NeosServer is alive"))

    def get_job_status(self, job_number: int, job_password: str) -> str:
        """
        Returns the status of the job.

        Parameters
        ----------
        job_number : int
        job_password : str

        Returns
        -------
        str
            Either "Done", "Running", "Waiting", "Unknown Job", or "Bad Password"
        """
        return self.neos.getJobStatus(job_number, job_password)

    def get_completion_code(self, job_number: int, job_password: str) -> str:
        """
        Gets the completion code for "Done" jobs. Result is undefined for jobs
        that are "Waiting" or "Running". Returns "Normal", "Out of memory", "Timed out",
        "Disk Space", "Server error", "Unknown Job", or "Bad Password"

        Parameters
        ----------
        job_number : int
        job_password : str

        Returns
        -------
        str
        """
        return self.neos.getCompletionCode(job_number, job_password)

    def get_job_info(self, job_number: int, job_password: str) -> tuple:
        """
        Gets information about the job.

        Parameters
        ----------
        job_number : int
        job_password : str

        Returns
        -------
        tuple
            (category, solver_name, input, status, completion_code)
        """
        return self.neos.getJobInfo(job_number, job_password)

    def kill_job(self, job_number: int, job_password: str, killmsg="") -> str:
        """
        Cancel a submitted job that is running or waiting to run on NEOS.

        Parameters
        ----------
        job_number : int
        job_password : str
        killmsg : str, optional

        Returns
        -------
        str
        """
        return self.neos.killJob(job_number, job_password, killmsg)

    def get_final_results(
        self, job_number: int, job_password: str, is_blocking: bool = True
    ) -> xmlrpc.client.Binary:
        """
        Retrieve results from a submitted job on NEOS. If the job is still
        running and the user is not authenticated, then this function will
        hang until the job is finished. The function returns a base-64 encoded object.

        Parameters
        ----------
        job_number : int
        job_password : str
        is_blocking : bool, optional

        Returns
        -------
        xmlrpc.client.Binary
        """
        if is_blocking:
            return self.neos.getFinalResults(job_number, job_password)

        return self.neos.getFinalResultsNonBlocking(job_number, job_password)

    def email_job_results(self, job_number: int, job_password: str) -> str:
        """
        Results for a finished job will be emailed to the email address specified in the job submission.
        If results are too large for email, they will not be emailed (though they can be accessed via
        the NEOS website)


        Parameters
        ----------
        job_number : int
        job_password : str

        Returns
        -------
        str
        """
        return self.neos.emailJobResults(job_number, job_password)

    def get_intermediate_results(
        self,
        job_number: int,
        job_password: str,
        offset: int,
        is_blocking: bool = True,
    ) -> xmlrpc.client.Binary:
        """
        Gets intermediate results of a job submitted to NEOS, starting at the specified character offset up to
        the last received data. Intermediate results are usually the standard output of the solver daemon.
        Note that because output does not stream for jobs with "long" priority (default value), getIntermediateResults()
        will not return any results for long priority jobs. Output does stream for jobs with "short" priority
        (maximum time of 5 minutes).

        Parameters
        ----------
        job_number : int
        job_password : str
        offset : int
        is_blocking : bool, optional

        Returns
        -------
        xmlrpc.client.Binary
        """
        if is_blocking:
            return self.neos.getIntermediateResults(
                job_number, job_password, offset
            )

        return self.neos.getIntermediateResultsNonBlocking(
            job_number, job_password
        )

    def download_output(
        self,
        job_number: int,
        job_password: str,
        working_directory: str = ".",
    ) -> None:
        """
        Downloads the output of the job and writes it in a zip file.

        Parameters
        ----------
        job_number : int
        job_password : str
        working_directory: str
        """
        filename = f"{job_number}-{job_password}-solver-output.zip"
        try:
            os.makedirs(working_directory)
        except FileExistsError:
            pass

        response = self.neos.getOutputFile(job_number, job_password, filename)
        if str(response) == "Output file does not exist":
            raise NeosClientException(
                "Couldn't get output file from NEOS Server because:"
                f" {response}"
            )

        with open(os.path.join(working_directory, filename), "wb") as file:
            file.write(response.data)

        with zipfile.ZipFile(
            os.path.join(working_directory, filename), "r"
        ) as zip_ref:
            zip_ref.extractall(working_directory)

    def _prepare_xml(
        self,
        gams_string: str,
        solver_options: dict | None,
        gdx_path: str,
        restart_path: str,
        options: Options,
        save_name: str | None = None,
        xml_path: str = "neos.xml",
        working_directory: str = ".",
    ) -> None:
        gdx_string = ""
        if os.path.exists(gdx_path):
            with open(gdx_path, "rb") as gdx_file:
                content = gdx_file.read()
                gdx_base64 = base64.b64encode(content).decode("utf-8")
                gdx_string = f"<base64>{gdx_base64}</base64>"

        restart_string = ""
        try:
            with open(restart_path, "rb") as restart_file:
                content = restart_file.read()
                restart_base64 = base64.b64encode(content).decode("utf-8")
                restart_string = f"<base64>{restart_base64}\n</base64>"
        except FileNotFoundError:
            pass

        solver_options_str = ""
        if solver_options:
            solver_options_str = "\n".join(
                [f"{key} {value}" for key, value in solver_options.items()]
            )

        options._export(os.path.join(working_directory, "parameters"))
        with open(
            os.path.join(working_directory, "parameters"), encoding="utf-8"
        ) as file:
            parameters = [line.rstrip() for line in file.readlines()]

        extras = []
        if save_name is not None:
            extras.append(f"save={save_name}")

        parameter_string = "\n".join(parameters + extras)

        template = f"""
            <document>
            <category>lp</category>
            <solver>cplex</solver>
            <inputType>GAMS</inputType>
            <email>{self.email}</email>
            <priority>{self.priority}</priority>
            <model><![CDATA[{gams_string}]]></model>
            <options><![CDATA[{solver_options_str}]]></options>
            <parameters><![CDATA[{parameter_string}]]></parameters>
            <gdx>{gdx_string}</gdx>
            <restart>{restart_string}</restart>
            <wantgdx><![CDATA[]]></wantgdx>
            <wantlst><![CDATA[yes]]></wantlst>
            <wantlog><![CDATA[yes]]></wantlog>
            </document>
        """

        with open(
            os.path.join(working_directory, xml_path), "w", encoding="utf-8"
        ) as neos_xml:
            neos_xml.write(template)

    def print_queue(self):
        """Prints NEOS Server queue"""
        if not self.is_alive():
            raise NeosClientException(
                "NeosServer is not alive. Try again later."
            )

        msg = self.neos.printQueue()
        print(msg)

    def submit_job(
        self,
        xml_path: str = "neos.xml",
        working_directory: str = ".",
    ) -> tuple[int, str]:
        """
        Submits the job to NEOS Server.

        Parameters
        ----------
        xml_path : str, optional
        is_blocking : bool, optional

        Returns
        -------
        Tuple[int, str]
            Job number and job password

        Raises
        ------
        NeosClientException
            In case there was an error on NeosServer
        """
        with open(
            os.path.join(working_directory, xml_path), encoding="utf-8"
        ) as file:
            xml = file.read()

        if not self.is_alive():
            raise NeosClientException(
                "NeosServer is not alive. Try again later."
            )

        if self.username is not None and self.password is not None:
            job_number, job_password = self.neos.authenticatedSubmitJob(
                xml, self.username, self.password
            )
        else:
            job_number, job_password = self.neos.submitJob(xml)

        logger.info(f"Job Number: {job_number}, Job Password: {job_password}")
        self.jobs.append((job_number, job_password))

        if job_number == 0:
            raise NeosClientException(f"NEOS Server error! {job_password}")

        return job_number, job_password


class NEOSServer(backend.Backend):
    def __init__(
        self,
        container: Container,
        options: Options,
        solver: str,
        solver_options: dict | None,
        client: NeosClient | None,
        output: io.TextIOWrapper | None,
        model: Model,
        load_symbols: list[Symbol] | None,
    ) -> None:
        if client is None or not isinstance(client, NeosClient):
            raise ValidationError(
                "`NeosClient` must be provided to solve on NEOS Server."
            )

        super().__init__(
            container,
            model,
            options,
            solver,
            solver_options,
            output,
            load_symbols,
        )
        self.client = client
        self.job_name = self.get_job_name()
        self.gms_file = self.job_name + ".gms"
        self.pf_file = self.job_name + ".pf"
        self.restart_file = self.job_name + ".g00"
        self.trace_file = self.job_name + ".txt"

    def is_async(self):
        return not self.client.is_blocking

    def run(
        self,
        relaxed_domain_mapping: bool = False,
        gams_to_gamspy: bool = False,
    ):
        # Run a dummy job to get the restart file to be sent to NEOS Server
        self._create_restart_file()

        # Generate gams string and write modified symbols to gdx
        gams_string = self.preprocess("in.gdx")

        # Run the model
        self.execute_gams(gams_string)

        if self.is_async():
            return None

        # Synchronize GAMSPy with checkpoint and return a summary
        summary = self.postprocess()

        # Run another dummy job to synchronize GAMS and GAMSPy state
        self._sync()

        return summary

    def execute_gams(self, gams_string: str):
        working_directory = self.container.working_directory
        if self.container._debugging_level == "keep":
            self.options.log_file = os.path.basename(self.job_name) + ".log"

        extra_options = {
            "gdx": "output.gdx",
            "gdxSymbols": "newOrChanged",
            "trace": os.path.basename(self.trace_file),
            "forcework": "1",
        }
        self.options._set_extra_options(extra_options)

        self.client._prepare_xml(
            gams_string,
            solver_options=self.solver_options,
            gdx_path=self.container._gdx_in,
            restart_path=self.restart_file,
            options=self.options,
            working_directory=working_directory,
        )

        job_number, job_password = self.client.submit_job(
            working_directory=working_directory
        )

        if self.client.is_blocking:
            _ = self.client.get_final_results(job_number, job_password)

            self.client.download_output(
                job_number,
                job_password,
                working_directory=working_directory,
            )

            out_gdx = os.path.join(working_directory, "output.gdx")
            log_path = os.path.join(working_directory, "solve.log")
            if not os.path.exists(out_gdx):
                raise GamspyException(
                    "The job was not completed successfully. Check"
                    f" {log_path} for details."
                )
            shutil.move(out_gdx, self.container._gdx_out)

            if self.output is not None:
                with open(log_path) as file:
                    self.output.write(file.read())

            if self.options.log_file is not None and not os.path.exists(
                self.options.log_file
            ):
                shutil.copy(log_path, self.options.log_file)

        self.container._unsaved_statements = []
        if not self.is_async() and self.model:
            self.model._update_model_attributes()
            self.container._delete_autogenerated_symbols()

    def postprocess(self):
        super().load_records()
        if self.client.is_blocking:
            self.parse_listings()
            return self.prepare_summary(self.trace_file)

        return None

    def _prepare_dummy_options(self) -> dict:
        scrdir = self.container._process_directory

        extra_options = {
            "gdx": self.container._gdx_out,
            "gdxSymbols": "newOrChanged",
            "trace": self.trace_file,
            "input": self.gms_file,
            "sysdir": self.container.system_directory,
            "scrdir": scrdir,
            "scriptnext": os.path.join(scrdir, "gamsnext.sh"),
            "previouswork": 1,
            "license": self.container._license_path,
        }

        if self.container._network_license:
            extra_options["netlicense"] = os.path.join(scrdir, "gamslice.dat")

        return extra_options

    def _create_restart_file(self):
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write("")

        options = Options()
        extra_options = self._prepare_dummy_options()
        options._set_extra_options(extra_options)
        options._extra_options["save"] = self.restart_file
        options._export(self.pf_file)

        self.container._send_job(self.job_name, self.pf_file)

    def _sync(self):
        symbols = utils._get_symbol_names_from_gdx(
            self.container.system_directory, self.container._gdx_out
        )
        dirty_str = ",".join(symbols)
        with open(self.gms_file, "w", encoding="utf-8") as gams_file:
            gams_file.write(
                f'execute_load "{self.container._gdx_out}", {dirty_str};'
            )

        options = Options()
        extra_options = self._prepare_dummy_options()
        options._set_extra_options(extra_options)
        options._export(self.pf_file)

        self.container._send_job(self.job_name, self.pf_file)
