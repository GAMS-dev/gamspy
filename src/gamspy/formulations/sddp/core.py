from __future__ import annotations

import re
import signal
import threading
import time
import warnings

import numpy as np
import pandas as pd

import gamspy as gp
from gamspy.exceptions import GamspyException, ValidationError
from gamspy.formulations.sddp.cut_selection import LastCuts
from gamspy.formulations.sddp.noise import NoiseConfig
from gamspy.formulations.sddp.policy import PolicyResult
from gamspy.formulations.sddp.result import SDDPResult, _sci
from gamspy.formulations.sddp.risk import CVaR
from gamspy.formulations.sddp.simulation import SimulationResult
from gamspy.formulations.sddp.state import StateVar


class SDDP:
    """
    Stochastic Dual Dynamic Programming for multistage stochastic GAMSPy
    models.

    SDDP trains a cost-to-go approximation for a multistage stochastic
    program by alternating forward simulation passes with backward Benders
    cut generation. Register the state variable(s) with ``add_state``, the
    stochastic noise with ``set_noise``, inject the algorithm into the model
    with ``build``, then ``train`` the policy.

    Parameters
    ----------
    container : Container
        The ``gp.Container`` holding every user-defined symbol.
    stage_set : Set
        The full stage set; its length is the number of stages in the problem.
    time_set : Set | None
        Full time-domain set when the model has a finer-grained time inside
        each stage. By default None, which reuses ``stage_set``.
    n_trials : int
        Number of trial levels per state variable (must be >= 1). By default 5.
    seed : int
        Seed for the forward-pass scenario sampler. By default 42.
    verbose : bool
        Print one convergence row per iteration during training. By default
        True.

    Examples
    --------
    >>> import numpy as np
    >>> import gamspy as gp
    >>> from gamspy.formulations.sddp import SDDP
    >>> m = gp.Container()
    >>> t = gp.Set(m, "t", records=["jan", "feb", "mar", "apr"])
    >>> sddp = SDDP(m, stage_set=t, n_trials=2, seed=42, verbose=False)
    >>> stage = sddp.active_stage
    >>> precip = gp.Parameter(m, "precip")
    >>> level = gp.Variable(m, "L", type="positive", domain=t)
    >>> spill = gp.Variable(m, "F", type="positive", domain=t)
    >>> shortfall = gp.Variable(m, "Z", type="positive", domain=t)
    >>> release = gp.Variable(m, "R", type="positive", domain=t)
    >>> cost = gp.Variable(m, "COST")
    >>> release.up[t] = 200.0
    >>> level.up[t] = 250.0
    >>> balance = gp.Equation(m, "balance", domain=t)
    >>> obj = gp.Equation(m, "obj")
    >>> balance[t].where[stage[t]] = (
    ...     level[t] - level[t.lag(1, "circular")]
    ...     + release[t] + spill[t] - shortfall[t] == precip
    ... )
    >>> obj[...] = cost == gp.Sum(stage[t], 10.0 * spill[t] + 5.0 * shortfall[t])
    >>> sddp.add_state(level, initial_state=100.0)
    >>> sddp.set_noise(precip, scenario_data=np.array([[50.0], [50.0], [-50.0], [-50.0]]))
    >>> sddp.build(stage_cost=cost)
    >>> sddp
    SDDP(stages=4, states=['L'], noise=precip, built=True)

    """

    _EPS = 1e-9

    def __init__(
        self,
        container: gp.Container,
        stage_set: gp.Set,
        time_set: gp.Set | None = None,
        n_trials: int = 5,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if not isinstance(container, gp.Container):
            raise ValidationError("container must be a gp.Container instance")
        if not isinstance(stage_set, gp.Set):
            raise ValidationError("stage_set must be a gp.Set instance")
        if time_set is not None and not isinstance(time_set, gp.Set):
            raise ValidationError("time_set must be a gp.Set instance or None")
        if n_trials < 1:
            raise ValidationError(f"n_trials must be >= 1, got {n_trials}")

        self._container: gp.Container = container
        self._stage_set: gp.Set = stage_set
        self._time_set: gp.Set = time_set if time_set is not None else stage_set
        self._n_trials: int = n_trials
        self._seed: int = seed
        self._verbose: bool = verbose

        # sddp-owned active-stage singleton; the user reads this back via
        # `sddp.active_stage` and references it in their .where[...] gates.
        self._active_stage_set: gp.Set = gp.Set(
            container,
            "sddp_active_stage",
            domain=stage_set,
            description="sddp-owned active-stage singleton",
        )

        self._states: list[StateVar] = []
        self._noise: NoiseConfig | None = None

        # Counter so each simulate() call can use unique GAMSPy symbol names.
        self._sim_call_count: int = 0

        # populated by build()
        self._built: bool = False
        self._iteration_set: gp.Set | None = None
        self._active_cut_iteration_set: gp.Set | None = None
        self._trial_set: gp.Set | None = None
        self._alpha: gp.Variable | None = None
        self._approx_cost: gp.Variable | None = None
        self._gp_model: gp.Model | None = None

        self._loaded_from_save: bool = False
        self._trained: bool = False

    @property
    def active_stage(self) -> gp.Set:
        """sddp-owned active-stage singleton.

        Reference this in your equations' ``.where[stage[...]]`` clauses so
        each per-stage solve activates only the equations for the current
        stage.
        """
        return self._active_stage_set

    @property
    def container(self) -> gp.Container:
        """
        The host ``gp.Container`` holding every symbol owned by this sddp instance.
        """
        return self._container

    def _resolve_report(self, report: list[gp.Variable] | None) -> list[gp.Variable]:
        """
        Normalize a ``report=`` argument for ``policy()`` / ``simulate()``.
        """
        if report is None:
            return [state_var.variable for state_var in self._states]

        resolved: list[gp.Variable] = []
        for item in report:
            if not isinstance(item, gp.Variable):
                raise ValidationError(
                    f"report items must be gp.Variable, got "
                    f"{type(item).__name__}. To reference a variable by name, "
                    f"pass the variable object, e.g. sddp.container['R']."
                )
            resolved.append(item)
        return resolved

    def _resolve_state(self, state: float | dict[str, float]) -> dict[str, float]:
        names = [state_var.variable.name for state_var in self._states]
        if isinstance(state, dict):
            expected = set(names)
            got = set(state)
            if got != expected:
                raise ValidationError(
                    f"state keys {sorted(got)} must match the state variables {names}."
                )
            return {n: float(state[n]) for n in names}

        if len(self._states) != 1:
            raise ValidationError(
                f"scalar state but {len(self._states)} states {names}; pass a dict keyed by state name."
            )
        return {names[0]: float(state)}

    def _restore_user_bounds(self) -> None:
        for (
            variable,
            variable_domain,
            lower_bound_snapshot,
            upper_bound_snapshot,
        ) in self._user_bound_snapshots:
            if not variable_domain:
                variable.lo[...] = lower_bound_snapshot[...]
                variable.up[...] = upper_bound_snapshot[...]
            elif len(variable_domain) == 1:
                variable.lo[variable_domain[0]] = lower_bound_snapshot[
                    variable_domain[0]
                ]
                variable.up[variable_domain[0]] = upper_bound_snapshot[
                    variable_domain[0]
                ]
            else:
                idx = tuple(variable_domain)
                variable.lo[idx] = lower_bound_snapshot[idx]
                variable.up[idx] = upper_bound_snapshot[idx]

    # registration

    def add_state(
        self,
        variable: gp.Variable,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        initial_state: float | None = None,
    ) -> None:
        """Register a state variable.

        Parameters
        ----------
        variable : Variable
            GAMSPy Variable for the state (reservoir level, inventory, etc.),
            indexed over the time set.
        lower_bound : float | None
            Lower end of the feasible range used to seed the initial uniform
            trial grid and to clamp the adaptive trial update. By default
            None, which infers the bound (see Notes).
        upper_bound : float | None
            Upper end of the feasible range, resolved like ``lower_bound``.
            By default None.
        initial_state : float | None
            Value the state takes before stage 1. By default None, which
            falls back to ``lower_bound``.

        Notes
        -----
        Each bound is resolved in the following order:

        1. Use the value passed here.
        2. If ``None``, read from ``variable.records["lower"]`` /
           ``variable.records["upper"]``.
        3. If the variable has no recorded bounds, fall back to the variable
           type's default (``positive`` gives ``(0, +inf)``, etc.).

        If a bound is passed explicitly while the variable also carries an
        explicit recorded bound that disagrees, a ``UserWarning`` is raised
        and the passed value still wins.
        """
        if self._built:
            raise ValidationError("Cannot call add_state() after build()")

        var_lo, var_up, src = self._infer_var_bounds(variable)

        # Resolve lower / upper with user-wins-with-warning semantics.
        if lower_bound is None:
            lo = var_lo
        else:
            lo = float(lower_bound)
            if src == "records" and abs(lo - var_lo) > 1e-12:
                warnings.warn(
                    f"add_state: user-supplied lower_bound={lo} for "
                    f"`{variable.name}` differs from the variable's recorded "
                    f"lower bound={var_lo}. Using {lo}. (Did you mean to set "
                    f"`{variable.name}.lo` differently, or omit lower_bound?)",
                    UserWarning,
                    stacklevel=2,
                )

        if upper_bound is None:
            up = var_up
        else:
            up = float(upper_bound)
            if src == "records" and abs(up - var_up) > 1e-12:
                warnings.warn(
                    f"add_state: user-supplied upper_bound={up} for "
                    f"`{variable.name}` differs from the variable's recorded "
                    f"upper bound={var_up}. Using {up}. (Did you mean to set "
                    f"`{variable.name}.up` differently, or omit upper_bound?)",
                    UserWarning,
                    stacklevel=2,
                )

        state_var = StateVar(  # type: ignore[call-arg]
            variable=variable,
            lower_bound=lo,
            upper_bound=up,
            initial_state=float(initial_state) if initial_state is not None else None,
        )
        state_var.validate()
        self._states.append(state_var)

    def set_noise(
        self,
        parameter: gp.Parameter,
        scenario_data: np.ndarray,
        probabilities: np.ndarray | list[float] | None = None,
    ) -> None:
        """Register the stochastic noise model.

        Parameters
        ----------
        parameter : Parameter
            GAMSPy Parameter that is overwritten before each LP solve with
            the sampled inflow value for the current scenario.
        scenario_data : np.ndarray
            2-D numpy array of shape ``(n_stages, n_scenarios)``.
        probabilities : np.ndarray | list[float] | None
            Optional 1-D array of scenario probabilities, shape
            ``(n_scenarios,)``. Must be non-negative and sum to 1.0. By
            default None, which makes the scenarios equally likely
            (probability ``1/n_scenarios`` each).
        """
        if self._built:
            raise ValidationError("Cannot call set_noise() after build()")
        if self._noise is not None:
            raise ValidationError("set_noise() already called")

        probability_array: np.ndarray | None = None
        if probabilities is not None:
            probability_array = np.asarray(probabilities, dtype=float)

        noise_config = NoiseConfig(
            parameter=parameter,
            scenario_data=np.asarray(scenario_data, dtype=float),
            probabilities=probability_array,
        )
        n_stages = (
            len(self._stage_set.records) if self._stage_set.records is not None else 0
        )
        if n_stages:
            noise_config.validate(n_stages)
        self._noise = noise_config

    # build

    def build(
        self,
        stage_cost: gp.Variable,
        equations: list | None = None,
    ) -> None:
        """Inject the SDDP algorithm into the user model.

        Parameters
        ----------
        stage_cost : Variable
            GAMSPy Variable equal to the per-stage operational cost
            (WITHOUT the future-cost alpha term).
        equations : list | None
            User physics equations to include in the LP. By default None,
            which makes the sddp module pull every equation currently
            declared in the container.
        """
        if self._built:
            raise ValidationError("build() already called")
        if not self._states:
            raise ValidationError("Call add_state() before build()")
        if self._noise is None:
            raise ValidationError("Call set_noise() before build()")

        if equations is None:
            equations = list(self._container.getEquations())
            if not equations:
                raise ValidationError(
                    "build(equations=None) found no equations in the container. "
                    "Either declare your equations before calling build(), or "
                    "pass them explicitly via build(equations=[...])."
                )

        # Pre-flight: at least one equation must reference stage_cost. Without
        # this the LP minimises a free variable that nothing constrains, so
        # the solve returns -inf (or a bound-constrained extreme) silently.
        if not self._equation_references_var(equations, stage_cost):
            raise ValidationError(
                "None of the supplied equations reference the stage-cost"
            )

        self._user_variables = list(self._container.getVariables())
        container = self._container
        active_stage_set = self._active_stage_set
        stage_set = self._stage_set
        time_set = self._time_set

        stage_labels = stage_set.toList()
        time_labels = time_set.toList()
        n_stages = len(stage_labels)
        n_times = len(time_labels)

        if n_times % n_stages != 0:
            raise ValidationError(
                f"len(time_set)={n_times} is not divisible by len(stage_set)={n_stages}"
            )

        # For example, hourly time data and weekly stages yield 168 steps/stage.
        time_steps_per_stage = n_times // n_stages

        stage_end_time: dict[str, str] = {}
        previous_stage_end_time: dict[str, str] = {}
        for stage_index, stage_label in enumerate(stage_labels):
            stage_end_time[stage_label] = time_labels[
                (stage_index + 1) * time_steps_per_stage - 1
            ]
            previous_stage_end_time[stage_label] = time_labels[
                ((stage_index - 1) % n_stages + 1) * time_steps_per_stage - 1
            ]

        stage_end_map = gp.Set(
            container,
            "sddp_stage_end",
            domain=[stage_set, time_set],
            records=[
                (stage_label, stage_end_time[stage_label])
                for stage_label in stage_labels
            ],
            description="last time step of each stage",
        )
        previous_stage_end_map = gp.Set(
            container,
            "sddp_previous_stage_end",
            domain=[stage_set, time_set],
            records=[
                (stage_label, previous_stage_end_time[stage_label])
                for stage_label in stage_labels
            ],
            description="last time step of the previous stage (circular)",
        )

        # Alias for time_set used inside equation/loop bodies whose control
        # index is the same underlying set as time_set (i.e. when
        # time_steps_per_stage == 1 and the stage set is also the time set).
        # GAMS requires distinct aliases to avoid the "already in control"
        # error.
        #
        # CONVENTION for future edits: any new equation or .where[...] gate
        # referencing the time-domain index must use `time_alias` (or
        # self._time_alias), NEVER `time_set`/`self._time_set`. Bypassing this
        # works for multi-step stages (hydro) but breaks one-step stages
        # (ClearLake) with a runtime "Set is already in control" ValidationError.
        time_alias = gp.Alias(container, "sddp_time_alias", alias_with=time_set)

        # Step 1: iteration and active-cut-iteration sets
        iteration_set = gp.Set(
            container,
            "sddp_iteration",
            description="SDDP iteration index",
        )  # records set in train()
        active_cut_iteration_set = gp.Set(
            container,
            "sddp_active_cut_iteration",
            domain=iteration_set,
            description="active cut iterations (grows after each iteration)",
        )
        self._iteration_set = iteration_set
        self._active_cut_iteration_set = active_cut_iteration_set

        # Step 2: trial grid and per-state cut parameters
        n_trials = self._n_trials
        trial_labels = [f"trial{k}" for k in range(1, n_trials + 1)]
        trial_set = gp.Set(
            container,
            "sddp_trial",
            records=trial_labels,
            description="trial levels",
        )
        self._trial_set = trial_set

        trial_fractions = [
            (trial_index / (n_trials - 1) if n_trials > 1 else 0.0)
            for trial_index in range(n_trials)
        ]

        for state_var in self._states:
            lb, ub = state_var.lower_bound, state_var.upper_bound
            rows: list[tuple] = []
            for stage_label in stage_labels:
                previous_time = previous_stage_end_time[stage_label]
                for trial_index, trial_label in enumerate(trial_labels):
                    val = lb + trial_fractions[trial_index] * (ub - lb)
                    if abs(val) < self._EPS:
                        val = self._EPS
                    rows.append((trial_label, previous_time, val))
            state_var.trial_values = gp.Parameter(
                container,
                f"sddp_trial_values_{state_var.name}",
                domain=[trial_set, time_set],
                records=pd.DataFrame(rows, columns=["i", "t", "value"]),
                description=f"trial reservoir levels for {state_var.name}",
            )

            state_var.cut_slope = gp.Parameter(
                container,
                f"sddp_cut_slope_{state_var.name}",
                domain=[iteration_set, trial_set, stage_set],
                description=f"Benders cut slope for {state_var.name}",
            )
        cut_intercept = gp.Parameter(
            container,
            "sddp_cut_intercept",
            domain=[iteration_set, trial_set, stage_set],
            description="Benders cut intercept (delta), shared across states",
        )
        self._cut_intercept = cut_intercept

        trial_labels = trial_set.toList()

        # Step 3: Scenario set
        assert self._noise is not None
        noise_config = self._noise
        scenario_labels = [f"s{k}" for k in range(1, noise_config.n_scenarios + 1)]
        noise_config.scenario_set = gp.Set(
            container,
            "sddp_scenario",
            records=scenario_labels,
            description="noise scenarios",
        )
        scenario_set = noise_config.scenario_set

        # Probability per scenario: uniform by default, user-supplied otherwise.
        if noise_config.probabilities is not None:
            probability_values = noise_config.probabilities
        else:
            probability_values = np.full(
                noise_config.n_scenarios, 1.0 / noise_config.n_scenarios
            )
        scenario_prob = gp.Parameter(
            container,
            "sddp_prob",
            domain=scenario_set,
            records=pd.DataFrame(
                [
                    (scenario_label, float(probability_value))
                    for scenario_label, probability_value in zip(
                        scenario_labels, probability_values, strict=True
                    )
                ],
                columns=["scenario", "value"],
            ),
            description="scenario probabilities (uniform by default)",
        )
        self._scenario_prob = scenario_prob

        # Step 4: future-cost approximation and approximate total cost.
        alpha = gp.Variable(
            container,
            "sddp_alpha",
            type="positive",
            domain=stage_set,
            description="future cost approximation, one value per stage",
        )
        approx_cost = gp.Variable(
            container,
            "sddp_approx_cost",
            description="total approximate cost: stage_cost + alpha[next stage]",
        )
        self._alpha = alpha
        self._approx_cost = approx_cost

        # Step 5: obj_approx
        # approx_cost == stage_cost + alpha[stage_set+1]
        # For the last stage, Ord(stage_set) < Card(stage_set) is False -> Sum contributes 0.
        obj_approx = gp.Equation(
            container,
            "sddp_obj_approx",
            description="approximate objective with future cost",
        )
        obj_approx[...] = approx_cost == stage_cost + gp.Sum(
            stage_set.where[
                active_stage_set[stage_set] & (gp.Ord(stage_set) < gp.Card(stage_set))
            ],
            alpha[stage_set.lead(1)],
        )

        # Step 6: Benders cuts
        # alpha[stage_set+1] - sum_s slope[active iteration, trial, stage_set+1]
        # * state_s[last_of_w] >= intercept[active iteration, trial, stage_set]
        # The slope term SUMS over states (a single supporting hyperplane per
        # cut, with one slope per state); the intercept is shared across states.
        cuts = gp.Equation(
            container,
            "sddp_cuts",
            domain=[iteration_set, trial_set, stage_set],
            description="Benders cuts on future cost",
        )
        slope_total = None
        for state_var in self._states:
            assert state_var.cut_slope is not None
            term = gp.Sum(
                time_alias.where[stage_end_map[stage_set, time_alias]],
                state_var.cut_slope[
                    active_cut_iteration_set, trial_set, stage_set.lead(1)
                ]
                * state_var.variable[time_alias],
            )
            slope_total = term if slope_total is None else slope_total + term
        assert slope_total is not None
        cuts[active_cut_iteration_set, trial_set, stage_set].where[
            active_stage_set[stage_set] & (gp.Ord(stage_set) < gp.Card(stage_set))
        ] = (
            alpha[stage_set.lead(1)] - slope_total
            >= cut_intercept[active_cut_iteration_set, trial_set, stage_set]
        )

        # Step 7: LP model
        solve_opts = gp.Options(
            equation_listing_limit=0,
            variable_listing_limit=0,
            report_solution=2,
            solve_link_type="memory",
            merge_strategy="clear",
        )
        gp_model = gp.Model(
            container,
            "sddp_model",
            problem="lp",
            equations=list(equations) + [obj_approx, cuts],
            sense=gp.Sense.MIN,
            objective=approx_cost,
        )

        # Step 8: Backward GUSS dict
        # For each (trial, scenario): fix every state at its trial level,
        # inject the scenario noise, then collect approx_cost.l and each state's
        # marginal at the previous stage boundary.
        backward_scenario_set = gp.Set(
            container,
            "sddp_backward_scenario",
            domain=[trial_set, scenario_set],
            records=[
                (trial_label, scenario_label)
                for trial_label in trial_labels
                for scenario_label in scenario_labels
            ],
            description="backward GUSS scenario set (trial x noise)",
        )
        guss_options = gp.Parameter(
            container,
            "sddp_guss_options",
            domain="*",
            records=[("SkipBaseCase", 1), ("LogOption", 1), ("UpdateType", 2)],
        )
        self._guss_options = guss_options
        backward_noise = gp.Parameter(
            container,
            "sddp_backward_noise",
            domain=[trial_set, scenario_set],
            description="scatter: noise per backward scenario",
        )
        backward_approx_cost = gp.Parameter(
            container,
            "sddp_backward_approx_cost",
            domain=[trial_set, scenario_set],
            description="extract: approx_cost.l per backward scenario",
        )

        for state_var in self._states:
            state_var.backward_fixed_state = gp.Parameter(
                container,
                f"sddp_backward_fixed_state_{state_var.name}",
                domain=[trial_set, scenario_set, time_set],
                description=f"scatter: fix {state_var.name} at trial level",
            )
            state_var.backward_state_marginal = gp.Parameter(
                container,
                f"sddp_backward_state_marginal_{state_var.name}",
                domain=[trial_set, scenario_set, time_set],
                description=f"extract: {state_var.name}.m[prev_last] = cut slope",
            )
            state_var.cut_slope_accumulator = gp.Parameter(
                container,
                f"sddp_cut_slope_accumulator_{state_var.name}",
                domain=trial_set,
                description=f"cut slope accumulator for {state_var.name}",
            )

        backward_scenario_dict = gp.GUSSScenarioDict(
            container, "sddp_backward_scenario_dict", backward_scenario_set
        )
        backward_scenario_dict.add_options(guss_options)
        backward_scenario_dict.add_param(noise_config.parameter, backward_noise)
        for state_var in self._states:
            assert state_var.backward_fixed_state is not None
            backward_scenario_dict.add_fixed(
                state_var.variable, state_var.backward_fixed_state
            )  # fix at trial level
        backward_scenario_dict.add_level(
            approx_cost, backward_approx_cost
        )  # collect approx_cost.l
        for state_var in self._states:
            assert state_var.backward_state_marginal is not None
            backward_scenario_dict.add_marginal(
                state_var.variable, state_var.backward_state_marginal
            )  # collect x_s.m

        # Scenario noise as a GAMSPy parameter for symbolic loop assignments.
        stage_scenario_noise = gp.Parameter(
            container,
            "sddp_stage_scenario_noise",
            domain=[stage_set, scenario_set],
            records=pd.DataFrame(
                [
                    (
                        stage_labels[stage_index],
                        scenario_label,
                        float(noise_config.scenario_data[stage_index, scenario_index]),
                    )
                    for stage_index in range(n_stages)
                    for scenario_index, scenario_label in enumerate(scenario_labels)
                ],
                columns=["stage", "scenario", "value"],
            ),
            description="noise realization for each stage and scenario",
        )

        # Cut intercept accumulator (shared: a single intercept per cut).
        cut_intercept_accumulator = gp.Parameter(
            container,
            "sddp_cut_intercept_accumulator",
            domain=trial_set,
            description="cut intercept accumulator (per trial point)",
        )

        # CVaR change-of-measure weight accumulators
        scenario_alias = gp.Alias(
            container, "sddp_scenario_alias", alias_with=scenario_set
        )
        cvar_tail_mass = gp.Parameter(
            container,
            "sddp_cvar_mass",
            domain=[trial_set, scenario_set],
            description="CVaR: probability mass of worse scenarios per trial",
        )
        cvar_weight = gp.Parameter(
            container,
            "sddp_cvar_weight",
            domain=[trial_set, scenario_set],
            description="CVaR: backward blended change-of-measure weight",
        )

        # Step 9: Forward GUSS dict
        # n_trials scenarios per stage: one sampled noise path per trial level.
        sampled_path = gp.Set(
            container,
            "sddp_sampled_path",
            domain=[iteration_set, stage_set, trial_set, scenario_set],
            description="pre-sampled noise scenario for each (iteration, stage, trial)",
        )  # Path determined in train()

        forward_scenario_set = gp.Set(
            container,
            "sddp_forward_scenario",
            domain=[trial_set, scenario_set],
            description="active forward scenarios for current stage",
        )
        forward_noise = gp.Parameter(
            container,
            "sddp_forward_noise",
            domain=[trial_set, scenario_set],
            description="scatter: noise per forward scenario",
        )
        forward_approx_cost = gp.Parameter(
            container,
            "sddp_forward_approx_cost",
            domain=[trial_set, scenario_set],
            description="extract: approx_cost.l per forward scenario",
        )
        forward_stage_cost = gp.Parameter(
            container,
            "sddp_forward_stage_cost",
            domain=[trial_set, scenario_set],
            description="extract: stage_cost.l (without alpha)",
        )
        for state_var in self._states:
            state_var.forward_fixed_state = gp.Parameter(
                container,
                f"sddp_forward_fixed_state_{state_var.name}",
                domain=[trial_set, scenario_set, time_set],
                description=f"scatter: fix {state_var.name} at forward state",
            )
            state_var.forward_state_level = gp.Parameter(
                container,
                f"sddp_forward_state_level_{state_var.name}",
                domain=[trial_set, scenario_set, time_set],
                description=f"extract: {state_var.name}.l after forward solve",
            )

        forward_scenario_dict = gp.GUSSScenarioDict(
            container, "sddp_forward_scenario_dict", forward_scenario_set
        )
        forward_scenario_dict.add_options(guss_options)
        forward_scenario_dict.add_param(noise_config.parameter, forward_noise)
        for state_var in self._states:
            assert state_var.forward_fixed_state is not None
            forward_scenario_dict.add_fixed(
                state_var.variable, state_var.forward_fixed_state
            )  # fix at forward state
        forward_scenario_dict.add_level(approx_cost, forward_approx_cost)
        forward_scenario_dict.add_level(
            stage_cost, forward_stage_cost
        )  # collect stage cost (for forward_path_cost)
        for state_var in self._states:
            assert state_var.forward_state_level is not None
            forward_scenario_dict.add_level(
                state_var.variable, state_var.forward_state_level
            )  # res.l (next stage)

        # Step 9b: Stage-1 wait-and-see GUSS dict
        # n_scenarios LPs in a single GUSS batch, one per stage-1 realization
        # of the noise.
        initial_stage_noise = gp.Parameter(
            container,
            "sddp_initial_stage_noise",
            domain=scenario_set,
            description="scatter: precip per stage-1 scenario",
        )
        initial_stage_approx_cost = gp.Parameter(
            container,
            "sddp_initial_stage_approx_cost",
            domain=scenario_set,
            description="extract: approx_cost.l per stage-1 scenario (LB)",
        )
        initial_stage_cost = gp.Parameter(
            container,
            "sddp_initial_stage_cost",
            domain=scenario_set,
            description="extract: stage_cost.l per stage-1 scenario",
        )
        for state_var in self._states:
            state_var.initial_stage_fixed_state = gp.Parameter(
                container,
                f"sddp_initial_stage_fixed_state_{state_var.name}",
                domain=[scenario_set, time_set],
                description=f"scatter: fix {state_var.name} at initial state for stage 1",
            )
            state_var.initial_stage_state_level = gp.Parameter(
                container,
                f"sddp_initial_stage_state_level_{state_var.name}",
                domain=[scenario_set, time_set],
                description=f"extract: {state_var.name}.l per stage-1 scenario",
            )

        initial_stage_scenario_dict = gp.GUSSScenarioDict(
            container, "sddp_initial_stage_scenario_dict", scenario_set
        )
        initial_stage_scenario_dict.add_options(guss_options)
        initial_stage_scenario_dict.add_param(
            noise_config.parameter, initial_stage_noise
        )
        for state_var in self._states:
            assert state_var.initial_stage_fixed_state is not None
            initial_stage_scenario_dict.add_fixed(
                state_var.variable, state_var.initial_stage_fixed_state
            )
        initial_stage_scenario_dict.add_level(approx_cost, initial_stage_approx_cost)
        initial_stage_scenario_dict.add_level(stage_cost, initial_stage_cost)
        for state_var in self._states:
            assert state_var.initial_stage_state_level is not None
            initial_stage_scenario_dict.add_level(
                state_var.variable, state_var.initial_stage_state_level
            )

        # Step 10: Loop sets and convergence bookkeeping
        backward_step_alias = gp.Alias(
            container, "sddp_backward_step_alias", alias_with=stage_set
        )
        backward_stage_alias = gp.Alias(
            container, "sddp_backward_stage_alias", alias_with=stage_set
        )
        forward_stage_alias = gp.Alias(
            container, "sddp_forward_stage_alias", alias_with=stage_set
        )

        # Pair the first through penultimate stage labels with the last through
        # second stage labels, producing the reverse traversal order.
        backward_stage_map = gp.Set(
            container,
            "sddp_backward_stage_map",
            domain=[backward_step_alias, backward_stage_alias],
            records=[
                (stage_labels[k], stage_labels[-(k + 1)]) for k in range(n_stages - 1)
            ],
            description="backward pass index map",
        )
        forward_stage_set = gp.Set(
            container,
            "sddp_forward_stage_set",
            domain=[forward_stage_alias],
            records=stage_labels[1:],
            description="forward pass stages (second through last)",
        )

        # Convergence and state-tracking parameters
        convergence_metrics = gp.Parameter(
            container,
            "sddp_convergence",
            domain=[iteration_set, "*"],
            description="convergence metrics per iteration",
        )
        forward_path_cost = gp.Parameter(
            container,
            "sddp_path_cost",
            domain=[iteration_set, trial_set],
            description="accumulated stage cost per forward path",
        )
        for state_var in self._states:
            state_var.current_forward_state = gp.Parameter(
                container,
                f"sddp_forward_state_{state_var.name}",
                domain=trial_set,
                description=f"{state_var.name} state at end of current forward stage",
            )
            state_var.forward_state_history = gp.Parameter(
                container,
                f"sddp_forward_state_history_{state_var.name}",
                domain=[iteration_set, stage_set, trial_set],
                description=f"{state_var.name} state history for adaptive trials",
            )

        # Adaptive-update support symbols
        adaptive_trial_map = gp.Set(
            container,
            "sddp_adaptive_trial_map",
            domain=[time_set, stage_set],
            records=[
                (previous_stage_end_time[stage_label], stage_label)
                for stage_label in stage_labels[1:]
            ],
            description="map previous-stage boundary times to downstream stages",
        )
        adaptive_trial_times = gp.Set(
            container,
            "sddp_adaptive_trial_times",
            domain=time_set,
            records=[
                previous_stage_end_time[stage_label] for stage_label in stage_labels[1:]
            ],
            description="time indices written by the adaptive trial update",
        )
        previous_iteration_indicator = gp.Parameter(
            container,
            "sddp_previous_iteration",
            domain=iteration_set,
            description="indicator (0/1) selecting the previous iteration for adaptive update",
        )

        # Store everything train() will need
        self._stage_labels = stage_labels
        self._stage_end_time = stage_end_time
        self._previous_stage_end_time = previous_stage_end_time
        self._stage_end_map = stage_end_map
        self._previous_stage_end_map = previous_stage_end_map
        self._scenario_labels = scenario_labels
        self._trial_labels = trial_labels
        self._stage_cost_var = stage_cost
        self._user_equations = list(equations)
        self._obj_approx_eq = obj_approx
        self._cuts_eq = cuts
        self._solve_opts = solve_opts
        self._gp_model = gp_model
        self._backward_noise = backward_noise
        self._backward_approx_cost = backward_approx_cost
        self._backward_scenario_dict = backward_scenario_dict
        self._stage_scenario_noise = stage_scenario_noise
        self._cut_intercept_accumulator = cut_intercept_accumulator
        self._scenario_alias = scenario_alias
        self._cvar_tail_mass = cvar_tail_mass
        self._cvar_weight = cvar_weight
        self._sampled_path = sampled_path
        self._forward_scenario_set = forward_scenario_set
        self._forward_noise = forward_noise
        self._forward_approx_cost = forward_approx_cost
        self._forward_stage_cost = forward_stage_cost
        self._forward_scenario_dict = forward_scenario_dict
        self._initial_stage_scenario_dict = initial_stage_scenario_dict
        self._initial_stage_noise = initial_stage_noise
        self._initial_stage_approx_cost = initial_stage_approx_cost
        self._initial_stage_cost = initial_stage_cost
        self._backward_step_alias = backward_step_alias
        self._backward_stage_alias = backward_stage_alias
        self._forward_stage_alias = forward_stage_alias
        self._backward_stage_map = backward_stage_map
        self._forward_stage_set = forward_stage_set
        self._convergence_metrics = convergence_metrics
        self._forward_path_cost = forward_path_cost
        self._adaptive_trial_map = adaptive_trial_map
        self._adaptive_trial_times = adaptive_trial_times
        self._previous_iteration_indicator = previous_iteration_indicator
        self._time_alias = time_alias

        self._initial_state_time = previous_stage_end_time[stage_labels[0]]

        # Snapshot each state's full user-set bound profile
        type_defaults = {
            "positive": (0.0, float("inf")),
            "negative": (float("-inf"), 0.0),
            "binary": (0.0, 1.0),
            "integer": (0.0, float("inf")),
        }
        for state_var in self._states:
            d_lo, d_up = type_defaults.get(
                getattr(state_var.variable, "type", "free"),
                (float("-inf"), float("inf")),
            )
            recs = state_var.variable.records
            lo_rows: list[tuple] = []
            up_rows: list[tuple] = []
            for tl in time_labels:
                lo, up = d_lo, d_up
                if recs is not None and len(recs) > 0 and "lower" in recs.columns:
                    domain_col = recs.columns[0]
                    match = recs[recs[domain_col].astype(str) == tl]
                    if len(match) > 0:
                        lo = float(match["lower"].iloc[0])
                        up = float(match["upper"].iloc[0])
                lo_rows.append((tl, lo))
                up_rows.append((tl, up))

            state_var.orig_lo_param = gp.Parameter(
                container,
                f"sddp_orig_lo_{state_var.name}",
                domain=time_set,
                records=pd.DataFrame(lo_rows, columns=["t", "value"]),
                description=f"snapshot of user lower bounds for {state_var.name}",
            )
            state_var.orig_up_param = gp.Parameter(
                container,
                f"sddp_orig_up_{state_var.name}",
                domain=time_set,
                records=pd.DataFrame(up_rows, columns=["t", "value"]),
                description=f"snapshot of user upper bounds for {state_var.name}",
            )

        value_cols = {"level", "marginal", "lower", "upper", "scale"}
        state_names = {s.name for s in self._states}
        self._user_bound_snapshots: list[tuple] = []
        for variable in self._user_variables:
            if variable.name in state_names:
                continue
            variable_records = variable.records
            if variable_records is None or len(variable_records) == 0:
                continue
            if (variable_records["lower"] == float("-inf")).all() and (
                variable_records["upper"] == float("inf")
            ).all():
                continue  # fully free variable
            variable_domain = list(variable.domain)
            domain_columns = [
                c for c in variable_records.columns if c not in value_cols
            ]
            lower_bound_snapshot = gp.Parameter(
                container,
                f"sddp_blo_{variable.name}",
                domain=variable_domain,
                records=variable_records[[*domain_columns, "lower"]].rename(
                    columns={"lower": "value"}
                ),
                description=f"snapshot of user lower bounds for {variable.name}",
            )
            upper_bound_snapshot = gp.Parameter(
                container,
                f"sddp_bup_{variable.name}",
                domain=variable_domain,
                records=variable_records[[*domain_columns, "upper"]].rename(
                    columns={"upper": "value"}
                ),
                description=f"snapshot of user upper bounds for {variable.name}",
            )
            self._user_bound_snapshots.append(
                (variable, variable_domain, lower_bound_snapshot, upper_bound_snapshot)
            )

        self._built = True

    @staticmethod
    def _infer_var_bounds(var: gp.Variable) -> tuple[float, float, str]:
        type_defaults = {
            "positive": (0.0, float("inf")),
            "negative": (float("-inf"), 0.0),
            "binary": (0.0, 1.0),
            "integer": (0.0, float("inf")),
        }
        default_lo, default_up = type_defaults.get(
            getattr(var, "type", "free"), (float("-inf"), float("inf"))
        )
        recs = var.records
        if recs is None or len(recs) == 0:
            return default_lo, default_up, "type_default"
        if "lower" not in recs.columns or "upper" not in recs.columns:
            return default_lo, default_up, "type_default"
        return float(recs["lower"].min()), float(recs["upper"].max()), "records"

    @staticmethod
    def _equation_references_var(equations: list, var: gp.Variable) -> bool:
        pattern = re.compile(rf"\b{re.escape(var.name)}\b")
        for eq in equations:
            text = ""
            for accessor in ("getDefinition", "gamsRepr", "latexRepr"):
                fn = getattr(eq, accessor, None)
                if callable(fn):
                    try:
                        text = str(fn())
                        break
                    except Exception:
                        continue
            if not text:
                text = str(eq)
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _cvar_backward_weights(
        risk: CVaR,
        cvar_tail_mass: gp.Parameter,
        cvar_weight: gp.Parameter,
        cost: gp.Parameter,
        scenario_prob: gp.Parameter,
        scenario_alias: gp.Alias,
        trial_set: gp.Set,
        scenario_set: gp.Set,
    ) -> None:

        cvar_tail_mass[trial_set, scenario_set] = gp.Sum(
            scenario_alias.where[
                (cost[trial_set, scenario_alias] > cost[trial_set, scenario_set])
                | (
                    (cost[trial_set, scenario_alias] >= cost[trial_set, scenario_set])
                    & (gp.Ord(scenario_alias) < gp.Ord(scenario_set))
                )
            ],
            scenario_prob[scenario_alias],
        )
        cvar_weight[trial_set, scenario_set] = (1.0 - risk.weight) * scenario_prob[
            scenario_set
        ] + (
            risk.weight
            * gp.math.Max(
                0.0,
                gp.math.Min(
                    scenario_prob[scenario_set],
                    risk.tail - cvar_tail_mass[trial_set, scenario_set],
                ),
            )
            / risk.tail
        )

    def _set_active_cuts(self, completed: int, keep_iter: int | None) -> None:
        """Point the active-cut set at a window of the completed iterations.

        Cuts are only ever deactivated, never deleted: ``cut_slope`` and
        ``cut_intercept`` keep every value they were given, and the active-cut
        iteration set decides which of them become LP rows. Restoring the whole
        pool is therefore a single set assignment rather than a recomputation.

        Parameters
        ----------
        completed : int
            Number of iterations finished so far, i.e. labels
            ``iteration1..iteration{completed}``.
        keep_iter : int | None
            Retain only the most recent ``keep_iter`` of them; ``None`` retains
            all of them.
        """
        assert self._iteration_set is not None
        assert self._active_cut_iteration_set is not None
        iteration_set = self._iteration_set
        active_cut_iteration_set = self._active_cut_iteration_set

        first = 1 if keep_iter is None else max(1, completed - keep_iter + 1)
        active_cut_iteration_set[iteration_set] = False
        active_cut_iteration_set[iteration_set].where[
            (gp.Ord(iteration_set) >= first) & (gp.Ord(iteration_set) <= completed)
        ] = True

    def _stage1_bound(self, slot: str) -> float:
        """Expected stage-1 cost under whichever cuts are currently active.

        This is what the lower bound *is*, so pricing a different cut pool needs
        only this one wait-and-see batch: the pool is being evaluated, not
        extended, so no backward pass is involved. Costs ``n_scenarios`` LPs
        against the several thousand a full iteration solves.
        """
        assert self._noise is not None
        assert self._gp_model is not None
        scenario_set = self._noise.scenario_set
        assert scenario_set is not None

        container = self._container
        stage_set = self._stage_set
        active_stage_set = self._active_stage_set
        time_set = self._time_set
        convergence_metrics = self._convergence_metrics
        initial_stage_label = self._stage_labels[0]

        # Same single-sync pattern as the training loop: batch every statement
        # and this solve into one GAMS job rather than a round-trip apiece.
        container._in_loop += 1
        try:
            active_stage_set[stage_set] = False
            active_stage_set[initial_stage_label] = True
            for state_var in self._states:
                assert state_var.initial_stage_fixed_state is not None
                assert state_var.initial_stage_state_level is not None
                assert state_var.orig_lo_param is not None
                assert state_var.orig_up_param is not None
                state_var.variable.lo[time_set] = state_var.orig_lo_param[time_set]
                state_var.variable.up[time_set] = state_var.orig_up_param[time_set]
                state_var.initial_stage_fixed_state[scenario_set, time_set] = 0
                state_var.initial_stage_fixed_state[
                    scenario_set, self._initial_state_time
                ] = max(
                    state_var.initial_state
                    if state_var.initial_state is not None
                    else state_var.lower_bound,
                    self._EPS,
                )
                state_var.initial_stage_state_level[scenario_set, time_set] = 0

            self._initial_stage_noise[scenario_set] = self._stage_scenario_noise[
                initial_stage_label, scenario_set
            ]
            self._initial_stage_approx_cost[scenario_set] = 0
            self._initial_stage_cost[scenario_set] = 0

            self._gp_model.solve(
                options=self._solve_opts, scenario=self._initial_stage_scenario_dict
            )

            convergence_metrics[slot, "lo_full"] = gp.Sum(
                scenario_set,
                self._scenario_prob[scenario_set]
                * self._initial_stage_approx_cost[scenario_set],
            )
        finally:
            container._in_loop -= 1
        container._synch_with_gams()

        return float(
            _parameter_values_for_iteration(convergence_metrics, slot).get(
                "lo_full", 0.0
            )
        )

    # training (running) the model
    def train(
        self,
        n_iter: int = 20,
        rel_tol: float | None = None,
        patience: int = 5,
        risk: CVaR | None = None,
        gap_paths: int = 0,
        cut_selection: LastCuts | None = None,
    ) -> SDDPResult:
        """Run the SDDP iteration loop and return convergence results.

        Parameters
        ----------
        n_iter : int
            Maximum number of SDDP iterations; training may stop earlier when
            ``rel_tol`` is set. By default 20.
        rel_tol : float | None
            Relative lower-bound improvement below which an iteration counts
            as a plateau step. By default None.
        patience : int
            Number of consecutive sub-``rel_tol`` iterations required before
            stopping early (must be >= 1). Ignored when ``rel_tol`` is
            ``None``. By default 5.
        risk : CVaR | None
            Risk measure to optimize. By default None, the risk-neutral
            expectation.
        gap_paths : int
            If ``>= 1``, after training run an out-of-sample Monte-Carlo
            simulation of the trained policy with this many independent paths
            and report a statistically meaningful optimality gap (95% CI of the
            policy cost vs. the lower bound) on the result. By default 0.
        cut_selection : LastCuts | None
            Strategy bounding how many cuts stay active in the stage
            subproblems. By default None, keeping every cut.

        Notes
        -----
        Pressing CTRL+C during training stops gracefully: the current iteration
        is finalized (or, if its in-flight solve was aborted, discarded) and the
        policy trained so far is returned with ``stop_reason == "interrupted"``.
        Press CTRL+C a second time to abort hard (raises ``KeyboardInterrupt``).

        """
        if not self._built:
            raise ValidationError("Call build() before train()")
        if self._loaded_from_save:
            raise ValidationError(
                "train() cannot be called on a loaded sddp instance. "
                "Loaded instances are read-only; use policy() / simulate() "
                "for inference, or retrain from scratch."
            )

        for _name, _value in (
            ("n_iter", n_iter),
            ("patience", patience),
            ("gap_paths", gap_paths),
        ):
            if isinstance(_value, bool) or not isinstance(_value, int):
                raise ValidationError(f"{_name} must be an int")
        if n_iter < 1:
            raise ValidationError(f"n_iter must be >= 1, got {n_iter}")
        if rel_tol is not None and rel_tol <= 0:
            raise ValidationError(f"rel_tol must be > 0 when set, got {rel_tol}")
        if patience < 1:
            raise ValidationError(f"patience must be >= 1, got {patience}")
        if risk is not None and not isinstance(risk, CVaR):
            raise ValidationError("risk must be a CVaR(...) instance or None")
        if gap_paths < 0:
            raise ValidationError(f"gap_paths must be >= 0, got {gap_paths}")
        if cut_selection is not None and not isinstance(cut_selection, LastCuts):
            raise ValidationError(
                "cut_selection must be a LastCuts(...) instance or None"
            )
        if cut_selection is not None and cut_selection.keep_iter >= n_iter:
            warnings.warn(
                f"cut_selection=LastCuts(keep_iter={cut_selection.keep_iter}) "
                f"retains at least as many iterations as the {n_iter} about to "
                f"run, so no cut will ever be deactivated and cut selection "
                f"will have no effect.",
                UserWarning,
                stacklevel=2,
            )
            cut_selection = None

        if self._trained:
            raise ValidationError(
                "train() has already been called on this sddp instance."
            )
        self._trained = True

        # build() invariants: narrow Optional fields for the type checker.
        assert self._noise is not None
        assert self._iteration_set is not None
        assert self._active_cut_iteration_set is not None
        assert self._alpha is not None
        assert self._approx_cost is not None
        assert self._gp_model is not None

        container = self._container
        active_stage_set = self._active_stage_set
        stage_set = self._stage_set
        time_set = self._time_set

        noise_config = self._noise
        assert noise_config.scenario_set is not None
        trial_set = self._trial_set
        assert trial_set is not None
        scenario_set = noise_config.scenario_set
        iteration_set = self._iteration_set
        active_cut_iteration_set = self._active_cut_iteration_set

        stage_labels = self._stage_labels
        trial_labels = self._trial_labels
        scenario_labels = self._scenario_labels

        stage_end_map = self._stage_end_map
        previous_stage_end_map = self._previous_stage_end_map

        gp_model = self._gp_model
        solve_opts = self._solve_opts
        backward_scenario_dict = self._backward_scenario_dict
        forward_scenario_dict = self._forward_scenario_dict
        scenario_prob = self._scenario_prob

        backward_noise = self._backward_noise
        backward_approx_cost = self._backward_approx_cost
        cut_intercept_accumulator = self._cut_intercept_accumulator
        cut_intercept = self._cut_intercept
        scenario_alias = self._scenario_alias
        cvar_tail_mass = self._cvar_tail_mass
        cvar_weight = self._cvar_weight

        forward_scenario_set = self._forward_scenario_set
        forward_noise = self._forward_noise
        forward_approx_cost = self._forward_approx_cost
        forward_stage_cost = self._forward_stage_cost

        initial_stage_scenario_dict = self._initial_stage_scenario_dict
        initial_stage_noise = self._initial_stage_noise
        initial_stage_approx_cost = self._initial_stage_approx_cost
        initial_stage_cost = self._initial_stage_cost

        backward_step_alias = self._backward_step_alias
        backward_stage_alias = self._backward_stage_alias
        forward_stage_alias = self._forward_stage_alias
        backward_stage_map = self._backward_stage_map
        forward_stage_set = self._forward_stage_set
        sampled_path = self._sampled_path
        stage_scenario_noise = self._stage_scenario_noise

        convergence_metrics = self._convergence_metrics
        forward_path_cost = self._forward_path_cost
        adaptive_trial_map = self._adaptive_trial_map
        adaptive_trial_times = self._adaptive_trial_times
        previous_iteration_indicator = self._previous_iteration_indicator
        time_alias = self._time_alias

        initial_state_time = self._initial_state_time

        # Size the iteration machinery for this run
        iteration_set.setRecords([f"iteration{k}" for k in range(1, n_iter + 1)])
        iteration_labels = iteration_set.toList()

        rng = np.random.default_rng(seed=self._seed)
        sample_probabilities = (
            noise_config.probabilities
        )  # None -> uniform; preserves seed sequences
        sampled_path.setRecords(
            [
                (
                    iteration_label,
                    stage_label,
                    trial_label,
                    rng.choice(scenario_labels, p=sample_probabilities),
                )
                for iteration_label in iteration_labels
                for stage_label in stage_labels
                for trial_label in trial_labels
            ]
        )

        # CTRL+C handling
        self._stop_requested = False
        on_main_thread = threading.current_thread() is threading.main_thread()
        prev_sigint = signal.getsignal(signal.SIGINT)  # GAMSPy's handler

        def _sddp_sigint(signum, frame):
            if not self._stop_requested:
                self._stop_requested = True
                print(
                    "\n[sddp] interrupted by user - finishing the current "
                    "iteration; press CTRL+C again to abort."
                )
                return
            # Second CTRL+C: restore the original handler and hard-abort.
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
            raise KeyboardInterrupt

        def _restore_sigint():
            if on_main_thread and prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)

        if on_main_thread:
            signal.signal(signal.SIGINT, _sddp_sigint)

        result = SDDPResult()
        result.risk = risk
        result.cut_selection = cut_selection
        t_total = time.perf_counter()
        prev_lo: float | None = None
        best_windowed: float | None = None
        best_full: float | None = None
        flat_streak = 0  # consecutive iterations with sub-rel_tol LB gain
        completed = 0  # fully finished iterations (-> result.iterations_run)
        lo = up = up_95 = sigma = float("nan")  # last completed iteration's stats

        for iteration_index, iteration_label in enumerate(iteration_labels):
            # A CTRL+C during the previous iteration's tail -> stop cleanly here.
            if self._stop_requested:
                result.stop_reason = "interrupted"
                break

            previous_iteration_label = (
                iteration_labels[iteration_index - 1] if iteration_index > 0 else None
            )
            t_iter = time.perf_counter()
            interrupted = False

            # Single-sync pattern: pre-bump prevents nested Loop.__exit__ syncs.
            # One explicit _synch_with_gams() at the end covers the full iteration.
            container._in_loop += 1
            try:
                ##########          Adaptive trial-level update          ##########
                # Replace the static uniform grid with the previous iteration's
                # forward-pass reservoir states. The whole update is a single
                # symbolic GAMS assignment; the `previous_iteration_indicator` parameter selects
                # which iteration's frs to pull from, and the `adaptive_trial_map` set
                # threads (previous_stage_end_time[stage_label], stage_label) so the inner Sum has exactly
                # one non-zero term per (i, t). Clamp + eps floor are done
                # in-line via gp.math.Max/Min so we still get only one GAMS
                # statement (vs. ~n_trials x (n_stages - 1) per-iter before).
                if previous_iteration_label is not None:
                    previous_iteration_indicator[iteration_set] = 0
                    previous_iteration_indicator[previous_iteration_label] = 1
                    for state_var in self._states:
                        assert state_var.trial_values is not None
                        assert state_var.forward_state_history is not None
                        state_var.trial_values[trial_set, time_alias].where[
                            adaptive_trial_times[time_alias]
                        ] = gp.math.Max(
                            self._EPS,
                            gp.math.Min(
                                state_var.upper_bound,
                                gp.math.Max(
                                    state_var.lower_bound,
                                    gp.Sum(
                                        gp.Domain(iteration_set, stage_set).where[
                                            previous_iteration_indicator[iteration_set]
                                            & adaptive_trial_map[time_alias, stage_set]
                                        ],
                                        state_var.forward_state_history[
                                            iteration_set, stage_set, trial_set
                                        ],
                                    ),
                                ),
                            ),
                        )

                # Reset per-iteration accumulators
                forward_path_cost[iteration_label, trial_set] = 0
                for state_var in self._states:
                    assert state_var.current_forward_state is not None
                    assert state_var.forward_state_history is not None
                    state_var.current_forward_state[trial_set] = 0
                    state_var.forward_state_history[
                        iteration_label, stage_set, trial_set
                    ] = 0

                # Backward pass
                with gp.Loop(
                    gp.Domain(backward_step_alias, backward_stage_alias).where[
                        backward_stage_map[backward_step_alias, backward_stage_alias]
                    ]
                ):
                    # Activate the current backward stage
                    active_stage_set[stage_set] = False
                    active_stage_set[backward_stage_alias] = True

                    # Restore user bounds + scatter each state's trial level.
                    # GUSS leaves a state variable fixed at the prior batch's
                    # last scenario, which would pin the current stage's free
                    # decision variable at a stale value.
                    for state_var in self._states:
                        assert state_var.backward_fixed_state is not None
                        assert state_var.trial_values is not None
                        assert state_var.orig_lo_param is not None
                        assert state_var.orig_up_param is not None
                        state_var.variable.lo[time_set] = state_var.orig_lo_param[
                            time_set
                        ]
                        state_var.variable.up[time_set] = state_var.orig_up_param[
                            time_set
                        ]
                        # Scatter: fix x_s[prev_last_of_wloop] at trial level[i]
                        state_var.backward_fixed_state[
                            trial_set, scenario_set, time_set
                        ] = 0
                        state_var.backward_fixed_state[
                            trial_set, scenario_set, time_alias
                        ].where[
                            previous_stage_end_map[backward_stage_alias, time_alias]
                        ] = state_var.trial_values[trial_set, time_alias]

                    # Scatter: inject inflow scenario for this stage (same for all i)
                    backward_noise[trial_set, scenario_set] = stage_scenario_noise[
                        backward_stage_alias, scenario_set
                    ]

                    # GUSS solve: (n_trials x n_scenarios) LPs in one batch
                    gp_model.solve(options=solve_opts, scenario=backward_scenario_dict)

                    # Change-of-measure weight per scenario: nominal probability,
                    # or the CVaR Rockafellar-Uryasev weight when risk-averse.
                    if isinstance(risk, CVaR):
                        self._cvar_backward_weights(
                            risk,
                            cvar_tail_mass,
                            cvar_weight,
                            backward_approx_cost,
                            scenario_prob,
                            scenario_alias,
                            trial_set,
                            scenario_set,
                        )
                        weight = cvar_weight[trial_set, scenario_set]
                    else:
                        weight = scenario_prob[scenario_set]

                    # Cut slope, per state: weighted x_s.m[prev_last] over scenarios.
                    for state_var in self._states:
                        assert state_var.cut_slope_accumulator is not None
                        assert state_var.backward_state_marginal is not None
                        state_var.cut_slope_accumulator[trial_set] = gp.Sum(
                            gp.Domain(scenario_set, time_alias).where[
                                previous_stage_end_map[backward_stage_alias, time_alias]
                            ],
                            weight
                            * state_var.backward_state_marginal[
                                trial_set, scenario_set, time_alias
                            ],
                        )

                    # Cut intercept (shared): V(trial) - sum_s slope_s * trial_s.
                    # Uses LP strong duality:
                    # delta = E[approx_cost.l] - sum_s slope_s*x0_s.
                    slope_dot_trial = None
                    for state_var in self._states:
                        assert state_var.cut_slope_accumulator is not None
                        assert state_var.trial_values is not None
                        term = state_var.cut_slope_accumulator[trial_set] * gp.Sum(
                            time_alias.where[
                                previous_stage_end_map[backward_stage_alias, time_alias]
                            ],
                            state_var.trial_values[trial_set, time_alias],
                        )
                        slope_dot_trial = (
                            term if slope_dot_trial is None else slope_dot_trial + term
                        )
                    assert slope_dot_trial is not None
                    cut_intercept_accumulator[trial_set] = (
                        gp.Sum(
                            scenario_set,
                            weight * backward_approx_cost[trial_set, scenario_set],
                        )
                        - slope_dot_trial
                    )

                    # Store cut: per-state slope, shared intercept; activate active_cut_iteration_set.
                    for state_var in self._states:
                        assert state_var.cut_slope is not None
                        assert state_var.cut_slope_accumulator is not None
                        state_var.cut_slope[
                            iteration_label, trial_set, backward_stage_alias
                        ] = state_var.cut_slope_accumulator[trial_set]
                    cut_intercept[
                        iteration_label, trial_set, backward_stage_alias.lag(1)
                    ] = cut_intercept_accumulator[trial_set]
                    active_cut_iteration_set[iteration_label] = True

                # Forward stage 1: wait-and-see GUSS batch
                active_stage_set[stage_set] = False
                active_stage_set[stage_labels[0]] = True

                # Restore user bounds + fix each state at its initial level.
                # GUSS leaves variables at the prior batch's last scenario, same
                # as the backward sweep.
                for state_var in self._states:
                    assert state_var.initial_stage_fixed_state is not None
                    assert state_var.initial_stage_state_level is not None
                    assert state_var.orig_lo_param is not None
                    assert state_var.orig_up_param is not None
                    state_var.variable.lo[time_set] = state_var.orig_lo_param[time_set]
                    state_var.variable.up[time_set] = state_var.orig_up_param[time_set]
                    state_var.initial_stage_fixed_state[scenario_set, time_set] = 0
                    state_var.initial_stage_fixed_state[
                        scenario_set, initial_state_time
                    ] = max(
                        state_var.initial_state
                        if state_var.initial_state is not None
                        else state_var.lower_bound,
                        self._EPS,
                    )
                    state_var.initial_stage_state_level[scenario_set, time_set] = 0

                # Scatter: precip per scenario (shared)
                initial_stage_noise[scenario_set] = stage_scenario_noise[
                    stage_labels[0], scenario_set
                ]
                initial_stage_approx_cost[scenario_set] = 0
                initial_stage_cost[scenario_set] = 0

                gp_model.solve(options=solve_opts, scenario=initial_stage_scenario_dict)

                convergence_metrics[iteration_label, "lo"] = gp.Sum(
                    scenario_set,
                    scenario_prob[scenario_set]
                    * initial_stage_approx_cost[scenario_set],
                )

                # Per-trial stage-1 cost: pick each trial's pre-sampled scenario
                forward_path_cost[iteration_label, trial_set] = gp.Sum(
                    scenario_set.where[
                        sampled_path[
                            iteration_label, stage_labels[0], trial_set, scenario_set
                        ]
                    ],
                    initial_stage_cost[scenario_set],
                )

                # Per-trial forward state per state: x_s[last_of_w1] from the
                # trial's sampled scenario.
                for state_var in self._states:
                    assert state_var.current_forward_state is not None
                    assert state_var.forward_state_history is not None
                    assert state_var.initial_stage_state_level is not None
                    state_var.current_forward_state[trial_set] = gp.Sum(
                        gp.Domain(scenario_set, time_alias).where[
                            sampled_path[
                                iteration_label,
                                stage_labels[0],
                                trial_set,
                                scenario_set,
                            ]
                            & stage_end_map[stage_labels[0], time_alias]
                        ],
                        state_var.initial_stage_state_level[scenario_set, time_alias],
                    )
                    state_var.forward_state_history[
                        iteration_label, stage_labels[1], trial_set
                    ] = state_var.current_forward_state[trial_set]

                # Forward pass:
                with gp.Loop(
                    forward_stage_alias.where[forward_stage_set[forward_stage_alias]]
                ):
                    active_stage_set[stage_set] = False
                    active_stage_set[forward_stage_alias] = True

                    # Select the pre-sampled inflow scenario for each trial point
                    forward_scenario_set[trial_set, scenario_set] = False
                    forward_scenario_set[trial_set, scenario_set].where[
                        sampled_path[
                            iteration_label,
                            forward_stage_alias,
                            trial_set,
                            scenario_set,
                        ]
                    ] = True

                    # Scatter: inject inflow from sampled scenario
                    forward_noise[trial_set, scenario_set] = 0
                    forward_noise[trial_set, scenario_set].where[
                        forward_scenario_set[trial_set, scenario_set]
                    ] = stage_scenario_noise[forward_stage_alias, scenario_set]

                    # Per state: restore bounds, fix x_s[prev_last] at the forward
                    # state, reset its level extract.
                    for state_var in self._states:
                        assert state_var.forward_fixed_state is not None
                        assert state_var.forward_state_level is not None
                        assert state_var.current_forward_state is not None
                        assert state_var.orig_lo_param is not None
                        assert state_var.orig_up_param is not None
                        state_var.variable.lo[time_set] = state_var.orig_lo_param[
                            time_set
                        ]
                        state_var.variable.up[time_set] = state_var.orig_up_param[
                            time_set
                        ]
                        state_var.forward_fixed_state[
                            trial_set, scenario_set, time_set
                        ] = 0
                        state_var.forward_fixed_state[
                            trial_set, scenario_set, time_alias
                        ].where[
                            forward_scenario_set[trial_set, scenario_set]
                            & previous_stage_end_map[forward_stage_alias, time_alias]
                        ] = state_var.current_forward_state[trial_set]
                        state_var.forward_state_level[
                            trial_set, scenario_set, time_set
                        ] = 0

                    # Reset shared extract containers
                    forward_approx_cost[trial_set, scenario_set] = 0
                    forward_stage_cost[trial_set, scenario_set] = 0

                    # GUSS solve
                    gp_model.solve(options=solve_opts, scenario=forward_scenario_dict)

                    # Accumulate stage cost into forward_path_cost
                    forward_path_cost[iteration_label, trial_set] = forward_path_cost[
                        iteration_label, trial_set
                    ] + gp.Sum(
                        scenario_set.where[
                            forward_scenario_set[trial_set, scenario_set]
                        ],
                        forward_stage_cost[trial_set, scenario_set],
                    )

                    # Advance each state for the next stage
                    for state_var in self._states:
                        assert state_var.forward_state_level is not None
                        assert state_var.current_forward_state is not None
                        assert state_var.forward_state_history is not None
                        state_var.current_forward_state[trial_set] = gp.Sum(
                            gp.Domain(scenario_set, time_alias).where[
                                forward_scenario_set[trial_set, scenario_set]
                                & stage_end_map[forward_stage_alias, time_alias]
                            ],
                            state_var.forward_state_level[
                                trial_set, scenario_set, time_alias
                            ],
                        )
                        state_var.forward_state_history[
                            iteration_label, forward_stage_alias.lead(1), trial_set
                        ] = state_var.current_forward_state[trial_set]

                # Upper bound
                # Average total-path cost across all n_trials forward trajectories
                convergence_metrics[iteration_label, "up"] = gp.Sum(
                    trial_set, forward_path_cost[iteration_label, trial_set]
                ) / gp.Card(trial_set)

                # Cut selection: retire the iteration that has just fallen out
                # of the window.
                if cut_selection is not None:
                    expire = iteration_index - cut_selection.keep_iter
                    if expire >= 0:
                        active_cut_iteration_set[iteration_labels[expire]] = False
            except GamspyException:
                if self._stop_requested:
                    result.stop_reason = "interrupted"
                    interrupted = True
                else:
                    _restore_sigint()
                    raise
            finally:
                container._in_loop -= 1

            # Single explicit sync, now that the pre-bump is released so it runs
            # at the baseline _in_loop and actually flushes. Wrapped so a CTRL+C
            # that aborts the sync's own GAMS job is finalized too.
            if not interrupted:
                try:
                    container._synch_with_gams()
                except GamspyException:
                    if self._stop_requested:
                        result.stop_reason = "interrupted"
                        interrupted = True
                    else:
                        _restore_sigint()
                        raise

            if interrupted:
                break

            # Safety: the active-cut iteration set must round-trip back to
            # Python after every iteration. If it does not, downstream
            # policy()/simulate() solves see the set empty,
            # the Benders cuts deactivate, alpha collapses to its zero bound,
            # and decisions are silently wrong. Catches breakage of the
            # end-of-iteration sync at the moment it happens, not three steps
            # downstream when a user sees a nonsense policy result.
            active_labels = (
                0
                if active_cut_iteration_set.records is None
                else len(active_cut_iteration_set.records)
            )
            expected_labels = (
                min(iteration_index + 1, cut_selection.keep_iter)
                if cut_selection is not None
                else iteration_index + 1
            )
            if active_labels != expected_labels:
                _restore_sigint()
                raise GamspyException(
                    f"active-cut iteration set out of step after "
                    f"{iteration_label}: expected "
                    f"{expected_labels} active cut iteration(s), got "
                    f"{active_labels}. Either the end-of-iteration sync did not "
                    f"bring the active-cut iteration set back to Python, or "
                    f"cut selection retired the wrong label."
                )

            active_cuts = active_labels * self._n_trials

            # Read back convergence bounds
            # Filter for this iteration's rows
            convergence_values = _parameter_values_for_iteration(
                convergence_metrics, iteration_label
            )
            lo = convergence_values.get("lo", 0.0)
            elapsed = time.perf_counter() - t_iter

            # Forward-pass cost estimate from this iteration's trial paths
            # forward_path_cost[iteration_label, trial_label] = total cost on path trial_label.
            path_cost_values = _parameter_values_for_iteration(
                forward_path_cost, iteration_label
            )
            path_costs = np.array(
                [path_cost_values.get(trial_label, 0.0) for trial_label in trial_labels]
            )
            n_paths = len(path_costs)
            up = float(np.mean(path_costs))
            sigma = float(np.std(path_costs, ddof=1)) if n_paths > 1 else 0.0
            up_95 = up + 1.96 * sigma / np.sqrt(n_paths)

            # Cut selection makes the bound non-monotone by design: retiring a
            # binding cut may lower it.
            if cut_selection is None and prev_lo is not None and lo < prev_lo - 1e-3:
                _restore_sigint()
                raise GamspyException(
                    f"Lower bound decreased at {iteration_label}: {prev_lo:,.3f} -> {lo:,.3f}"
                )

            # Every `lo` is the optimum of a relaxation over a subset of valid
            # cuts, so each one bounds the true optimum from below and so does
            # their maximum.
            best_windowed = lo if best_windowed is None else max(best_windowed, lo)

            # LB-plateau
            if rel_tol is not None and prev_lo is not None:
                rel_gain = (lo - prev_lo) / max(abs(prev_lo), 1e-12)
                flat_streak = flat_streak + 1 if rel_gain < rel_tol else 0

            prev_lo = lo

            row = {
                "iteration": iteration_label,
                "lo": lo,
                "up": up,
                "sigma": sigma,
                "up_95": up_95,
                "elapsed": elapsed,
                "active_cuts": active_cuts,
            }
            result.convergence_table.append(row)

            if self._verbose:
                print(
                    f"  {iteration_label:>12s}:  "
                    f"bound = {_sci(lo):>14s}   "
                    f"sim cost = {_sci(up):>14s} ± {_sci(sigma):>12s}   "
                    f"[{elapsed:.1f}s]"
                )

            completed += 1

            # A CTRL+C landed during this iteration but the solve still finished
            if self._stop_requested:
                result.stop_reason = "interrupted"
                break

            # Stop once the LB has plateaued for `patience` consecutive iters
            if rel_tol is not None and flat_streak >= patience:
                converged = True
                if cut_selection is not None:
                    # A plateau under cut selection is ambiguous: the run may
                    # have converged, or the cuts retired earlier may have been
                    # the ones holding the bound up. Price the whole pool - one
                    # stage-1 batch - and only believe the plateau if the full
                    # pool agrees there is no progress left.
                    self._set_active_cuts(completed, None)
                    full_lo = self._stage1_bound(iteration_label)
                    self._set_active_cuts(completed, cut_selection.keep_iter)

                    # Measure against the previous full-pool reading once there
                    # is one
                    ref = best_full if best_full is not None else best_windowed
                    assert ref is not None  # set on every completed iteration
                    best_full = (
                        full_lo if best_full is None else max(best_full, full_lo)
                    )
                    if (full_lo - ref) / max(abs(ref), 1e-12) > rel_tol:
                        flat_streak = 0
                        converged = False
                        if self._verbose:
                            print(
                                f"  {iteration_label:>12s}:  plateau overruled - full "
                                f"cut pool bound = {_sci(full_lo)} beats the "
                                f"previous best {_sci(ref)}; continuing"
                            )
                if converged:
                    result.stop_reason = "converged"
                    break

        _restore_sigint()
        result.iterations_run = completed

        # Cut selection is a training-time device only. Restore every cut that
        # was ever generated, so the policy that is kept - and save(), policy()
        # and simulate() with it - uses the full approximation, and so the
        # reported bound is measured against the model actually being kept
        # rather than against a window that no longer exists. Runs on the
        # interrupt path too: an interrupted run must not hand back a windowed
        # policy.
        if cut_selection is not None and completed > 0:
            self._set_active_cuts(completed, None)
            try:
                lo = self._stage1_bound(iteration_labels[completed - 1])
            except GamspyException:
                # An interrupted run can leave GAMS unable to solve again. The
                # restoration above is still queued and will be applied by the
                # next GAMS interaction, so the policy is intact; only this
                # bound measurement is lost.
                if result.stop_reason != "interrupted":
                    raise
            else:
                # A superset of valid cuts cannot give a looser bound. This is
                # the invariant that replaces the monotonicity guard, and it is
                # sharper: it tests the thing that can actually go wrong.
                proved = [b for b in (best_windowed, best_full) if b is not None]
                witness = max(proved) if proved else None
                if witness is not None and lo < witness - 1e-6 * max(abs(witness), 1.0):
                    raise GamspyException(
                        f"the full cut pool gave a looser bound ({lo:,.6f}) than a "
                        f"bound already proved during training ({witness:,.6f}); a "
                        f"superset of valid cuts cannot do that, so cut selection "
                        f"retired or restored the wrong labels. "
                        f"Please report if you see this error"
                    )

        total = time.perf_counter() - t_total
        result.lower_bound = lo
        result.upper_bound = up
        result.upper_bound_95 = up_95
        result.sigma = sigma
        result.total_time = total

        # Optional end-of-training optimality-gap estimate.
        if gap_paths >= 1 and completed > 0 and result.stop_reason != "interrupted":
            self._sim_call_count += 1
            gap_sim = self._run_simulation(
                gap_paths,
                self._resolve_report(None),
                self._seed + 997,
                self._sim_call_count,
                quiet=True,
            )
            costs = gap_sim.total_cost.to_numpy(dtype=float)
            result.policy_cost_mean = float(costs.mean())
            result.policy_cost_stderr = (
                float(costs.std(ddof=1) / np.sqrt(len(costs)))
                if len(costs) > 1
                else 0.0
            )
            result.policy_cost_paths = int(gap_paths)

        if self._verbose:
            print(f"\nTotal: {total:.1f}s  ({total / max(completed, 1):.1f}s/iter avg)")
            print(f"\n{result}")

        return result

    # persistence (save / load)

    def save(self, path: str) -> None:
        """Serialize this sddp instance to a single ``.sddp`` file.

        The saved artifact contains the host ``Container`` (via
        ``gp.serialize``) plus a small JSON sidecar mapping symbol names
        to sddp roles. The output is loadable in a different Python
        process / notebook with ``SDDP.load(path)`` and supports
        ``policy()`` / ``simulate()`` immediately; it does not support
        further training (see ``SDDP.load`` for the rationale).

        Parameters
        ----------
        path : str
            Output path. Must end with ``.sddp``.

        Raises
        ------
        ValidationError
            If the instance was not built, or ``path`` does not end
            with ``.sddp``.
        """
        if not path.endswith(".sddp"):
            raise ValidationError(f"path must end with .sddp but found {path}")
        if not self._built:
            raise ValidationError(
                "Cannot save() a sddp instance that was not built and trained."
            )

        from gamspy.formulations.sddp.persistence import (
            _collect_sddp_metadata,
            _pack,
        )

        metadata = _collect_sddp_metadata(self)
        _pack(self._container, metadata, path)

    @classmethod
    def load(cls, path: str) -> SDDP:
        """Load an sddp instance.

        The returned instance is **read-only**: ``policy()`` and
        ``simulate()`` work as expected, but ``add_state()`` /
        ``set_noise()`` / ``build()`` / ``train()`` will raise. To add
        more iterations, retrain from scratch.

        Parameters
        ----------
        path : str
            Path to a ``.sddp`` file produced by ``SDDP.save()``.

        Returns
        -------
        SDDP
            An sddp instance reattached to the deserialized Container.

        Raises
        ------
        ValidationError
            If ``path`` does not end with ``.sddp``, the file is missing,
            malformed, references symbols absent from the Container, or
            carries a major version different from the current sddp module.
        """
        if not path.endswith(".sddp"):
            raise ValidationError(f"path must end with .sddp but found {path}")

        from gamspy.formulations.sddp.persistence import (
            _reattach_sddp,
            _unpack,
            _validate_metadata,
        )

        container, metadata = _unpack(path)
        _validate_metadata(metadata)
        return _reattach_sddp(container, metadata)

    # simulate

    def simulate(
        self,
        n_paths: int = 100,
        report: list[gp.Variable] | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        """Run the trained policy on fresh Monte Carlo paths.

        Parameters
        ----------
        n_paths : int
            Number of independent simulation paths. By default 100.
        report : list[Variable] | None
            GAMSPy Variables to capture per (path, stage). By default None,
            which captures every state variable.
        seed : int | None
            Sampler seed. By default None, which sets it to ``train_seed + 1``.

        Returns
        -------
        SimulationResult
            Pivot-shaped DataFrames (paths x stages) for total cost, stage
            costs, realised noise, and each reported variable.
        """
        if not self._built:
            raise ValidationError("Call build() before simulate()")
        if n_paths < 1:
            raise ValidationError(f"n_paths must be >= 1, got {n_paths}")

        resolved = self._resolve_report(report)
        for variable in resolved:
            if list(variable.domain) != [self._time_set]:
                raise NotImplementedError(
                    f"simulate() v1 only captures variables with domain==[time_set]; "
                    f"got {variable.name} with domain={variable.domain}"
                )

        if seed is None:
            seed = self._seed + 1

        self._sim_call_count += 1
        return self._run_simulation(n_paths, resolved, seed, self._sim_call_count)

    def _run_simulation(
        self,
        n_paths: int,
        report: list[gp.Variable],
        seed: int,
        call_id: int,
        quiet: bool = False,
    ) -> SimulationResult:
        start_time = time.perf_counter()

        # build() invariants: narrow Optional fields for the type checker.
        assert self._noise is not None
        assert self._active_cut_iteration_set is not None
        assert self._gp_model is not None
        assert self._approx_cost is not None

        container = self._container
        noise_config = self._noise
        assert noise_config.scenario_set is not None
        scenario_set = noise_config.scenario_set
        stage_set = self._stage_set
        time_set = self._time_set
        time_alias = self._time_alias
        active_stage_set = self._active_stage_set
        stage_labels = self._stage_labels
        scenario_labels = self._scenario_labels
        stage_end_map = self._stage_end_map
        previous_stage_end_map = self._previous_stage_end_map
        initial_state_time = self._initial_state_time
        stage_scenario_noise = self._stage_scenario_noise

        # Every state variable's post-solve level drives forward propagation, so
        # it must be extracted even if the caller's `report` omits it.
        state_vars = [state_var.variable for state_var in self._states]
        report_names = {variable.name for variable in report}
        extract_vars = list(report) + [
            variable for variable in state_vars if variable.name not in report_names
        ]

        suffix = f"_{call_id}"

        # Build per-call simulation symbols
        path_labels = [f"p{k}" for k in range(1, n_paths + 1)]
        sim_paths = gp.Set(
            container,
            f"sddp_sim_p{suffix}",
            records=path_labels,
            description="simulation paths",
        )

        rng = np.random.default_rng(seed)
        sample_probabilities = (
            noise_config.probabilities
        )  # None -> uniform; preserves seed sequences
        sample_records = [
            (
                path_label,
                stage_label,
                rng.choice(scenario_labels, p=sample_probabilities),
            )
            for path_label in path_labels
            for stage_label in stage_labels
        ]
        sim_sample = gp.Set(
            container,
            f"sddp_sim_sample{suffix}",
            domain=[sim_paths, stage_set, scenario_set],
            records=sample_records,
            description="pre-sampled noise scenario per (path, stage)",
        )

        # Active (path, scenario) tuples for the current stage, mutated in loop.
        sim_active = gp.Set(
            container,
            f"sddp_sim_active{suffix}",
            domain=[sim_paths, scenario_set],
            description="active simulation scenarios for current stage",
        )

        # Scatter parameters
        sim_noise = gp.Parameter(
            container,
            f"sddp_sim_noise{suffix}",
            domain=[sim_paths, scenario_set],
            description="scatter: noise per simulation scenario",
        )
        sim_state: dict[str, gp.Parameter] = {}
        for state_var in self._states:
            sim_state[state_var.variable.name] = gp.Parameter(
                container,
                f"sddp_sim_state_{state_var.name}{suffix}",
                domain=[sim_paths, scenario_set, time_set],
                description=f"scatter: fix {state_var.name} per simulation scenario",
            )

        # Extract parameters
        sim_approx_cost = gp.Parameter(
            container,
            f"sddp_sim_approx_cost{suffix}",
            domain=[sim_paths, scenario_set],
            description="extract: approx_cost.l per simulation scenario",
        )
        sim_cost = gp.Parameter(
            container,
            f"sddp_sim_cost{suffix}",
            domain=[sim_paths, scenario_set],
            description="extract: stage_cost.l per simulation scenario",
        )
        sim_var_level: dict[str, gp.Parameter] = {}
        for variable in extract_vars:
            sim_var_level[variable.name] = gp.Parameter(
                container,
                f"sddp_sim_var_{variable.name}{suffix}",
                domain=[sim_paths, scenario_set, time_set],
                description=f"extract: {variable.name}.l per simulation scenario",
            )

        # Per-stage history (path x stage)
        sim_cost_hist = gp.Parameter(
            container,
            f"sddp_sim_cost_hist{suffix}",
            domain=[sim_paths, stage_set],
            description="stage cost per (path, stage)",
        )
        sim_noise_hist = gp.Parameter(
            container,
            f"sddp_sim_noise_hist{suffix}",
            domain=[sim_paths, stage_set],
            description="realised noise per (path, stage)",
        )
        sim_var_hist: dict[str, gp.Parameter] = {}
        for variable in report:
            sim_var_hist[variable.name] = gp.Parameter(
                container,
                f"sddp_sim_{variable.name}_hist{suffix}",
                domain=[sim_paths, stage_set],
                description=f"history: {variable.name}.l at last_of_stage per (path, stage)",
            )

        # Forward state tracker per state (path -> end-of-previous-stage level)
        sim_fwd_state: dict[str, gp.Parameter] = {}
        for state_var in self._states:
            sim_fwd_state[state_var.variable.name] = gp.Parameter(
                container,
                f"sddp_sim_fwd_state_{state_var.name}{suffix}",
                domain=sim_paths,
                description=f"end-of-previous-stage {state_var.name} per path",
            )

        # GUSS dict for the simulation
        sim_dict = gp.GUSSScenarioDict(
            container,
            f"sddp_sim_dict{suffix}",
            sim_active,
        )
        sim_dict.add_options(self._guss_options)
        sim_dict.add_param(noise_config.parameter, sim_noise)
        for state_var in self._states:
            sim_dict.add_fixed(state_var.variable, sim_state[state_var.variable.name])
        sim_dict.add_level(self._approx_cost, sim_approx_cost)
        sim_dict.add_level(self._stage_cost_var, sim_cost)
        for variable in extract_vars:
            sim_dict.add_level(variable, sim_var_level[variable.name])

        # Forward simulation: one GUSS batch per stage
        container._in_loop += 1
        try:
            for stage_index, stage_label in enumerate(stage_labels):
                # Activate this stage
                active_stage_set[stage_set] = False
                active_stage_set[stage_label] = True

                # Build active (path, scenario) set for this stage
                sim_active[sim_paths, scenario_set] = sim_sample[
                    sim_paths, stage_label, scenario_set
                ]

                # Scatter noise: only the sampled scenario per path is non-zero
                sim_noise[sim_paths, scenario_set] = 0
                sim_noise[sim_paths, scenario_set].where[
                    sim_active[sim_paths, scenario_set]
                ] = stage_scenario_noise[stage_label, scenario_set]

                # Per state: restore bounds and fix the incoming state.
                for state_var in self._states:
                    assert state_var.orig_lo_param is not None
                    assert state_var.orig_up_param is not None
                    state_var.variable.lo[time_set] = state_var.orig_lo_param[time_set]
                    state_var.variable.up[time_set] = state_var.orig_up_param[time_set]

                    state_param = sim_state[state_var.variable.name]
                    state_param[sim_paths, scenario_set, time_set] = 0
                    if stage_index == 0:
                        initial = (
                            state_var.initial_state
                            if state_var.initial_state is not None
                            else state_var.lower_bound
                        )
                        state_param[sim_paths, scenario_set, initial_state_time].where[
                            sim_active[sim_paths, scenario_set]
                        ] = max(initial, self._EPS)
                    else:
                        state_param[sim_paths, scenario_set, time_alias].where[
                            sim_active[sim_paths, scenario_set]
                            & previous_stage_end_map[stage_label, time_alias]
                        ] = sim_fwd_state[state_var.variable.name][sim_paths]

                # Reset extracts
                sim_approx_cost[sim_paths, scenario_set] = 0
                sim_cost[sim_paths, scenario_set] = 0
                for variable in extract_vars:
                    sim_var_level[variable.name][sim_paths, scenario_set, time_set] = 0

                # GUSS solve: n_paths LPs in one batch
                self._gp_model.solve(options=self._solve_opts, scenario=sim_dict)

                # Record per-stage history
                sim_cost_hist[sim_paths, stage_label] = gp.Sum(
                    scenario_set.where[sim_active[sim_paths, scenario_set]],
                    sim_cost[sim_paths, scenario_set],
                )
                sim_noise_hist[sim_paths, stage_label] = gp.Sum(
                    scenario_set.where[sim_active[sim_paths, scenario_set]],
                    sim_noise[sim_paths, scenario_set],
                )
                for variable in report:
                    sim_var_hist[variable.name][sim_paths, stage_label] = gp.Sum(
                        gp.Domain(scenario_set, time_alias).where[
                            sim_active[sim_paths, scenario_set]
                            & stage_end_map[stage_label, time_alias]
                        ],
                        sim_var_level[variable.name][
                            sim_paths, scenario_set, time_alias
                        ],
                    )

                # Advance each state's forward value for the next stage.
                if stage_index < len(stage_labels) - 1:
                    for state_var in self._states:
                        sim_fwd_state[state_var.variable.name][sim_paths] = gp.Sum(
                            gp.Domain(scenario_set, time_alias).where[
                                sim_active[sim_paths, scenario_set]
                                & stage_end_map[stage_label, time_alias]
                            ],
                            sim_var_level[state_var.variable.name][
                                sim_paths, scenario_set, time_alias
                            ],
                        )

        except Exception:
            container._in_loop -= 1
            raise
        container._in_loop -= 1

        # Single sync flushes the loop. The per-stage history params assigned
        # above flag themselves for load-back, so the container loads them
        # automatically.
        container._synch_with_gams()

        # Aggregate into DataFrames
        cost_df = self._pivot_history(sim_cost_hist, path_labels, stage_labels)
        noise_df = self._pivot_history(sim_noise_hist, path_labels, stage_labels)
        var_dfs = {
            variable.name: self._pivot_history(
                sim_var_hist[variable.name], path_labels, stage_labels
            )
            for variable in report
        }
        total_cost = cost_df.sum(axis=1)
        total_cost.name = "total_cost"

        elapsed = time.perf_counter() - start_time
        sim_result = SimulationResult(
            n_paths=n_paths,
            total_cost=total_cost,
            stage_costs=cost_df,
            noise=noise_df,
            variables=var_dfs,
            elapsed=elapsed,
        )

        if self._verbose and not quiet:
            print(f"\nsimulate: {sim_result}  [{elapsed:.1f}s]")

        return sim_result

    # policy point query
    def policy(
        self,
        stage: str,
        state: float | dict[str, float],
        noise: float,
        report: list[gp.Variable] | None = None,
    ) -> PolicyResult:
        """Query the trained policy at a single situation.

        Answers: *"I'm in `stage`, my state arrived at `state`, this stage's
        noise realised as `noise`. What is the optimal decision and what
        does it cost me from here on?"*

        Parameters
        ----------
        stage : str
            Stage label to stand in (must be one of the defined stages).
        state : float | dict[str, float]
            The state value(s) entering this stage. With a single registered
            state variable, pass a scalar (e.g. ``180``). With several states,
            pass a ``dict`` keyed by state-variable name, e.g.
            ``{"L_up": 120, "L_dn": 200}`` where its keys must match the registered
            states exactly. A scalar with multiple states raises ``ValidationError``.
        noise : float
            The realised noise value for this stage.
        report : list[Variable] | None
            Variables whose optimal level to return. By default None, which
            returns every state variable.

        Returns
        -------
        PolicyResult
            ``stage``, ``incoming_state`` (scalar for one state, ``dict`` for
            several), ``noise``, ``approx_cost_to_go`` and ``decisions``
            (``{var_name: level}``).
        """
        if not self._built:
            raise ValidationError("Call build() and train() before policy()")

        # build() invariants: narrow Optional fields for the type checker.
        assert self._noise is not None
        assert self._active_cut_iteration_set is not None
        assert self._gp_model is not None
        assert self._approx_cost is not None

        if stage not in self._stage_labels:
            raise ValidationError(
                f"stage {stage!r} is not a valid stage; "
                f"expected one of {self._stage_labels}"
            )

        if (
            self._active_cut_iteration_set.records is None
            or len(self._active_cut_iteration_set.records) == 0
        ):
            warnings.warn(
                "policy() called before train(); no cuts exist yet, so the "
                "future-cost approximation is absent and the returned decision "
                "is NOT the trained policy. Call train() first.",
                UserWarning,
                stacklevel=2,
            )

        noise_config = self._noise
        state_values = self._resolve_state(state)

        resolved = self._resolve_report(report)
        for variable in resolved:
            if not variable.domain or variable.domain[0] != self._time_set:
                raise NotImplementedError(
                    f"policy() requires variables whose first domain element "
                    f"is the time set; got {variable.name} with "
                    f"domain={variable.domain}"
                )

        active_stage_set = self._active_stage_set
        stage_set = self._stage_set
        time_set = self._time_set
        previous_stage_end_time = self._previous_stage_end_time[stage]
        stage_end_time = self._stage_end_time[stage]

        # Activate only this stage
        active_stage_set[stage_set] = False
        active_stage_set[stage] = True

        # Restore user bounds, then fix each state at its incoming value.
        self._restore_user_bounds()
        for state_var in self._states:
            assert state_var.orig_lo_param is not None
            assert state_var.orig_up_param is not None
            state_var.variable.lo[time_set] = state_var.orig_lo_param[time_set]
            state_var.variable.up[time_set] = state_var.orig_up_param[time_set]
            state_var.variable.fx[previous_stage_end_time] = max(
                state_values[state_var.variable.name], self._EPS
            )

        # Inject the realised noise
        noise_config.parameter[...] = float(noise)

        # Single plain solve
        self._gp_model.solve(options=self._solve_opts)

        cost_to_go = self._scalar_level(self._approx_cost)
        decisions = {
            variable.name: self._extract_decision(variable, stage_end_time)
            for variable in resolved
        }

        # Logical cleanup: unfix every state
        for state_var in self._states:
            assert state_var.orig_lo_param is not None
            assert state_var.orig_up_param is not None
            state_var.variable.lo[time_set] = state_var.orig_lo_param[time_set]
            state_var.variable.up[time_set] = state_var.orig_up_param[time_set]

        incoming: float | dict[str, float] = (
            state_values[self._states[0].variable.name]
            if len(self._states) == 1
            else state_values
        )
        result = PolicyResult(
            stage=stage,
            incoming_state=incoming,
            noise=float(noise),
            approx_cost_to_go=cost_to_go,
            decisions=decisions,
        )
        return result

    @staticmethod
    def _scalar_level(var: gp.Variable) -> float:
        rec = var.records
        if rec is None or len(rec) == 0:
            return 0.0
        return float(rec["level"].iloc[0])

    @staticmethod
    def _extract_decision(var: gp.Variable, t_label: str):
        """Pull a variable's level(s) at the given time label.

        Returns:
        - ``float`` for 1-D (time-only) variables.
        - ``dict[str, float]`` for 2-D variables (time + one other dim),
          keyed by the non-time dimension's label.
        - ``dict[tuple[str, ...], float]`` for 3+-D variables (time + n
          other dims), keyed by a tuple of the non-time dim labels in
          declaration order.
        """
        rec = var.records
        dim = var.dimension

        if rec is None or len(rec) == 0:
            return 0.0 if dim == 1 else {}

        tcol = rec.columns[0]
        match = rec[rec[tcol].astype(str) == t_label]
        if len(match) == 0:
            return 0.0 if dim == 1 else {}

        if dim == 1:
            return float(match["level"].iloc[0])

        # Multi-dim: filter to records at t_label, key by remaining dim values.
        value_cols = {"level", "marginal", "lower", "upper", "scale"}
        other_dim_cols = [c for c in rec.columns if c != tcol and c not in value_cols]

        if len(other_dim_cols) == 1:
            col = other_dim_cols[0]
            return {str(row[col]): float(row["level"]) for _, row in match.iterrows()}

        return {
            tuple(str(row[c]) for c in other_dim_cols): float(row["level"])
            for _, row in match.iterrows()
        }

    @staticmethod
    def _pivot_history(
        param: gp.Parameter,
        row_labels: list[str],
        col_labels: list[str],
    ) -> pd.DataFrame:
        rec = param.records
        if rec is None or len(rec) == 0:
            return pd.DataFrame(0.0, index=row_labels, columns=col_labels)
        cols = [c for c in rec.columns if c != "value"]
        pivot = rec.pivot(index=cols[0], columns=cols[1], values="value")
        pivot = pivot.reindex(index=row_labels, columns=col_labels)
        pivot = pivot.fillna(0.0)
        pivot.index.name = "path"
        pivot.columns.name = "stage"
        return pivot

    # helpers
    @property
    def n_stages(self) -> int:
        recs = self._stage_set.records
        return len(recs) if recs is not None else 0

    def __repr__(self) -> str:
        states = [state_var.name for state_var in self._states]
        noise = self._noise.parameter.name if self._noise else "none"
        return (
            f"SDDP(stages={self.n_stages}, "
            f"states={states}, noise={noise}, built={self._built})"
        )


def _parameter_values_for_iteration(
    parameter: gp.Parameter, iteration_label: str
) -> dict:
    records = parameter.records
    if records is None or len(records) == 0:
        return {}

    cols = [column for column in records.columns if column != "value"]
    if not cols:
        return {}

    mask = records[cols[0]].astype(str).values == iteration_label
    if not mask.any():
        return {}
    iteration_records = records.loc[mask]

    other = cols[1:]
    values = iteration_records["value"].astype(float).values

    if len(other) == 0:
        # Leading column was the only domain, so return a single-entry dict.
        return {(): float(values[0])}
    if len(other) == 1:
        keys = iteration_records[other[0]].astype(str).values
        return dict(zip(keys, values, strict=False))
    key_columns = [iteration_records[column].astype(str).values for column in other]
    return dict(zip(zip(*key_columns, strict=False), values, strict=False))
