from typing import Any, List, Optional, Union, TYPE_CHECKING
import gams.transfer as gt
import pandas as pd
import gamspy._algebra._expression as expression
import gamspy._algebra._operable as operable
import gamspy._algebra._condition as condition
import gamspy._symbols._implicits as implicits
import gamspy.utils as utils

if TYPE_CHECKING:  # pragma: no cover
    from gamspy import Set, Container


class Equation(gt.Equation, operable.OperableMixin):
    """
    Represents an Equation symbol in GAMS.
    https://www.gams.com/latest/docs/UG_Equations.html

    Parameters
    ----------
    container : Container
    name : str
    type : str
    domain : Union[Set, str], optional
    records : Any, optional
    domain_forwarding : bool, optional
    description : str, optional
    uels_on_axes: bool
    definition: Expression, optional
    definition_domain: list, optional

    Example
    ----------
    >>> m = gt.Container()
    >>> i = gt.Set(m, "i", records=['i1','i2'])
    >>> a = gt.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
    >>> e = gt.Equation(m, "e", [i])
    >>> e[i] = Sum(i, a[i]) == a
    """

    def __init__(
        self,
        container: "Container",
        name: str,
        type: str,
        domain: Optional[List[Union["Set", str]]] = None,
        records: Optional[Any] = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        definition: Optional[expression.Expression] = None,
        definition_domain: Optional[list] = None,
    ):
        super().__init__(
            container,
            name,
            type,
            domain,
            records,
            domain_forwarding,
            description,
            uels_on_axes,
        )

        # enable load on demand
        self._is_dirty = False

        # allow conditions
        self.where = condition.Condition(self)

        # add statement
        self.ref_container._addStatement(self)

        # add defition if exists
        self._definition_domain = definition_domain
        self.definition = definition

        # create attributes
        self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
        self._stage = self._create_attr("stage")
        self._range = self._create_attr("range")
        self._slacklo = self._create_attr("slacklo")
        self._slackup = self._create_attr("slackup")
        self._slack = self._create_attr("slack")
        self._infeas = self._create_attr("infeas")

    def _init_attributes(self) -> tuple:
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def _create_attr(self, attr_name):
        return implicits.ImplicitParameter(
            self.ref_container,
            name=f"{self.name}.{attr_name}",
            records=self.records,
        )

    @property
    def l(self):  # noqa: E741, E743
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def lo(self):
        return self._lo

    @property
    def up(self):
        return self._up

    @property
    def scale(self):
        return self._s

    @property
    def stage(self):
        return self._stage

    @property
    def range(self):
        return self._range

    @property
    def slacklo(self):
        return self._slacklo

    @property
    def slackup(self):
        return self._slackup

    @property
    def slack(self):
        return self._slack

    @property
    def infeas(self):
        return self._infeas

    @property
    def definition(self) -> Optional[expression.Expression]:
        return self._definition

    @definition.setter
    def definition(
        self, assignment: Optional[expression.Expression] = None
    ) -> None:
        """
        Needed for scalar equations
        >>> eq..  sum(wh,build(wh)) =l= 1;
        >>> eq.definition = Sum(wh, build[wh]) <= 1
        """
        if assignment is None:
            self._definition = assignment
            return

        eq_types = ["=e=", "=l=", "=g="]

        # In case of an MCP equation without any equality, add the equality
        if not any(eq_type in assignment.gamsRepr() for eq_type in eq_types):
            assignment = assignment == 0

        equation_map = {
            "nonbinding": "=n=",
            "external": "=x=",
            "cone": "=c=",
            "boolean": "=b=",
        }

        if self.type in equation_map.keys():
            assignment._op_type = equation_map[self.type]  # type: ignore

        domain = (
            self._definition_domain if self._definition_domain else self.domain
        )
        statement = expression.Expression(
            implicits.ImplicitEquation(
                self.ref_container,
                name=self.name,
                type=self.type,
                domain=domain,
            ),
            "..",
            assignment,
        )

        self.ref_container._addStatement(statement)
        self._definition = statement

    def __iter__(self):
        assert self._records is not None, (
            f"Equation {self.name} does not contain any records. Cannot"
            " iterate over an Equation with no records"
        )

        if self._records is not None:
            return self._records.iterrows()

    def __getitem__(self, indices: Union[list, str]):
        domain = utils._toList(indices)
        return implicits.ImplicitEquation(
            self.ref_container, name=self.name, type=self.type, domain=domain
        )

    def __setitem__(
        self,
        indices: Union[tuple, str, implicits.ImplicitSet],
        assignment: expression.Expression,
    ):
        if len(self._domain) == 0:
            raise Exception(
                "Cannot perform an indexed assignment over a scalar Equation."
                " Specify the domain of the equation to perform an indexed"
                " asssignment"
            )

        if isinstance(indices, (tuple, str)):
            if isinstance(indices, str):
                indices = [indices]  # type: ignore

            if len(self._domain) != len(indices):
                raise Exception(
                    "Dimension of the symbol domain and the dimension of the"
                    " assignment indices must be the same!\nEquation"
                    f" dimension: {len(self._domain)}\nIndexed assignment"
                    f" dimension: {len(indices)}"
                )
        else:
            if len(self._domain) != len(indices.domain):
                raise Exception(
                    "Dimension of the symbol domain and the dimension of the"
                    " assignment indices must be the same!\nEquation"
                    f" dimension: {len(self._domain)}\nIndexed assignment"
                    f" dimension: {len(indices.domain)}"
                )

        domain = utils._toList(indices)

        equation_map = {
            "nonbinding": "=n=",
            "external": "=x=",
            "cone": "=c=",
            "boolean": "=b=",
        }

        if self.type in equation_map.keys():
            assignment._op_type = equation_map[self.type]

        statement = expression.Expression(
            implicits.ImplicitEquation(
                self.ref_container,
                name=self.name,
                type=self.type,
                domain=domain,
            ),
            "..",
            assignment,
        )

        self.ref_container._addStatement(statement)
        self._is_dirty = True

    def __eq__(self, other):  # type: ignore
        return expression.Expression(self, "=e=", other)

    @property
    def records(self):
        if not self._is_dirty:
            return self._records

        updated_records = self.ref_container._loadOnDemand(self.name)
        if updated_records is not None:
            self.records = updated_records.copy()
            self.domain_labels = self.domain_names
        else:
            self.records = updated_records

        self._is_dirty = False

        return self._records

    @records.setter
    def records(self, records):
        if records is not None:
            if not isinstance(records, pd.DataFrame):
                raise TypeError("Symbol 'records' must be type DataFrame")

        # set records
        self._records = records

        self._requires_state_check = True
        self.modified = True

        self.ref_container._requires_state_check = True
        self.ref_container.modified = True

        if self._records is not None:
            if self.domain_forwarding:  # pragma: no cover
                self._domainForwarding()

                # reset state check flags for all symbols in the container
                for symnam, symobj in self.ref_container.data.items():
                    symobj._requires_state_check = True

    def gamsRepr(self) -> str:
        """
        Representation of this Equation in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getStatement(self) -> str:
        """
        Statement of the Equation declaration

        Returns
        -------
        str
        """
        output = f"Equation {self.name}"

        if self.domain:
            domain_names = [set.name for set in self.domain]  # type: ignore
            domain_str = "(" + ",".join(domain_names) + ")"
            output += domain_str

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"
        return output
