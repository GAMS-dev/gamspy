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
    >>> from gamspy.formulations import SDDP
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

        self._m: gp.Container = container
        self._stage_parent: gp.Set = stage_set
        self._time_set: gp.Set = time_set if time_set is not None else stage_set
        self._n_trials: int = n_trials
        self._seed: int = seed
        self._verbose: bool = verbose

        # sddp-owned active-stage singleton; the user reads this back via
        # `sddp.active_stage` and references it in their .where[...] gates.
        self._active_stage: gp.Set = gp.Set(
            container,
            "sddp_active",
            domain=stage_set,
            description="sddp-owned active-stage singleton",
        )

        self._states: list[StateVar] = []
        self._noise: NoiseConfig | None = None

        # Counter so each simulate() call can use unique GAMSPy symbol names.
        self._sim_call_count: int = 0

        # populated by build()
        self._built: bool = False
        self._j_set: gp.Set | None = None
        self._jj_set: gp.Set | None = None
        self._alpha: gp.Variable | None = None
        self._acost: gp.Variable | None = None
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
        return self._active_stage

    @property
    def container(self) -> gp.Container:
        """
        The host ``gp.Container`` holding every symbol owned by this sddp instance.
        """
        return self._m

    def _resolve_report(self, report: list[gp.Variable] | None) -> list[gp.Variable]:
        """
        Normalize a ``report=`` argument for ``policy()`` / ``simulate()``.
        """
        if report is None:
            return [sv.variable for sv in self._states]

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
        names = [sv.variable.name for sv in self._states]
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
        for v, vdom, lo_p, up_p in self._user_bound_snaps:
            if not vdom:
                v.lo[...] = lo_p[...]
                v.up[...] = up_p[...]
            elif len(vdom) == 1:
                v.lo[vdom[0]] = lo_p[vdom[0]]
                v.up[vdom[0]] = up_p[vdom[0]]
            else:
                idx = tuple(vdom)
                v.lo[idx] = lo_p[idx]
                v.up[idx] = up_p[idx]

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

        sv = StateVar(  # type: ignore[call-arg]
            variable=variable,
            lower_bound=lo,
            upper_bound=up,
            initial_state=float(initial_state) if initial_state is not None else None,
        )
        sv.validate()
        self._states.append(sv)

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

        prob_array: np.ndarray | None = None
        if probabilities is not None:
            prob_array = np.asarray(probabilities, dtype=float)

        nc = NoiseConfig(
            parameter=parameter,
            scenario_data=np.asarray(scenario_data, dtype=float),
            probabilities=prob_array,
        )
        n_stages = (
            len(self._stage_parent.records)
            if self._stage_parent.records is not None
            else 0
        )
        if n_stages:
            nc.validate(n_stages)
        self._noise = nc

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
            equations = list(self._m.getEquations())
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

        self._user_variables = list(self._m.getVariables())
        m = self._m
        ww = self._active_stage  # sddp-owned active-stage singleton
        w = self._stage_parent  # full stage set
        t_set = self._time_set

        w_labels = w.toList()
        t_labels = t_set.toList()
        n_stages = len(w_labels)
        n_times = len(t_labels)

        if n_times % n_stages != 0:
            raise ValidationError(
                f"len(time_set)={n_times} is not divisible by len(stage_set)={n_stages}"
            )

        hpw = n_times // n_stages  # time steps per stage (e.g. 168 hours/week)

        last_hour: dict[str, str] = {}
        prev_last_hour: dict[str, str] = {}
        for pos, wl in enumerate(w_labels):
            last_hour[wl] = t_labels[(pos + 1) * hpw - 1]
            prev_last_hour[wl] = t_labels[((pos - 1) % n_stages + 1) * hpw - 1]

        last_set = gp.Set(
            m,
            "sddp_last",
            domain=[w, t_set],
            records=[(wl, last_hour[wl]) for wl in w_labels],
            description="last time step of each stage",
        )
        prevlast_set = gp.Set(
            m,
            "sddp_prevlast",
            domain=[w, t_set],
            records=[(wl, prev_last_hour[wl]) for wl in w_labels],
            description="last time step of the previous stage (circular)",
        )

        # Alias for t_set used inside equation/loop bodies whose control index
        # is the same underlying set as t_set (i.e. when hpw == 1 and the
        # stage set is the same set as the time set).  GAMS requires distinct
        # aliases to avoid the "already in control" error.
        #
        # CONVENTION for future edits: any new equation or .where[...] gate
        # referencing the time-domain index must use `tt` (or self._tt),
        # NEVER `t_set`/`self._time_set`. Bypassing this works for hpw>1
        # problems (hydro) but breaks hpw=1 problems (ClearLake) with a
        # runtime "Set is already in control" ValidationError.
        tt = gp.Alias(m, "sddp_tt", alias_with=t_set)

        # Step 1: j and jj
        j = gp.Set(
            m,
            "sddp_j",
            description="SDDP iteration index",
        )  # records set in train()
        jj = gp.Set(
            m,
            "sddp_jj",
            domain=j,
            description="active cut iterations (subset of j, grows each iter)",
        )
        self._j_set = j
        self._jj_set = jj

        # Step 2: trial grid and per-state cut parameters
        n_trials = self._n_trials
        i_labels = [f"i{k}" for k in range(1, n_trials + 1)]
        i_set = gp.Set(
            m,
            "sddp_i",
            records=i_labels,
            description="trial levels",
        )
        self._i_set = i_set

        fracs = [
            (pos_i / (n_trials - 1) if n_trials > 1 else 0.0)
            for pos_i in range(n_trials)
        ]

        for sv in self._states:
            sv.trial_set = i_set

            lb, ub = sv.lower_bound, sv.upper_bound
            rows: list[tuple] = []
            for wl in w_labels:
                ph = prev_last_hour[wl]
                for pos_i, il in enumerate(i_labels):
                    val = lb + fracs[pos_i] * (ub - lb)
                    if abs(val) < self._EPS:
                        val = self._EPS
                    rows.append((il, ph, val))
            sv.trial_param = gp.Parameter(
                m,
                f"sddp_ires_{sv.name}",
                domain=[i_set, t_set],
                records=pd.DataFrame(rows, columns=["i", "t", "value"]),
                description=f"trial reservoir levels for {sv.name}",
            )

            sv.cut_slope = gp.Parameter(
                m,
                f"sddp_cm_{sv.name}",
                domain=[j, i_set, w],
                description=f"Benders cut slope (cont_m) for {sv.name}",
            )
        cut_intercept = gp.Parameter(
            m,
            "sddp_d",
            domain=[j, i_set, w],
            description="Benders cut intercept (delta), shared across states",
        )
        self._cut_intercept = cut_intercept

        sv = self._states[0]
        assert sv.trial_set is not None
        assert sv.cut_slope is not None
        i_labels = i_set.toList()

        # Step 3: Scenario set
        assert self._noise is not None
        nc = self._noise
        s_labels = [f"s{k}" for k in range(1, nc.n_scenarios + 1)]
        nc.scenario_set = gp.Set(
            m, "sddp_s", records=s_labels, description="inflow scenarios"
        )
        s_set = nc.scenario_set

        # Probability per scenario: uniform by default, user-supplied otherwise.
        if nc.probabilities is not None:
            prob_values = nc.probabilities
        else:
            prob_values = np.full(nc.n_scenarios, 1.0 / nc.n_scenarios)
        prob_param = gp.Parameter(
            m,
            "sddp_prob",
            domain=s_set,
            records=pd.DataFrame(
                [(sl, float(pv)) for sl, pv in zip(s_labels, prob_values, strict=True)],
                columns=["s", "value"],
            ),
            description="scenario probabilities (uniform by default)",
        )
        self._prob_param = prob_param

        # Step 4: alpha (Approcimation of future cost from the current stage) and acost (Total approximate cost)
        alpha = gp.Variable(
            m,
            "sddp_alpha",
            type="positive",
            domain=w,
            description="future cost approximation, one value per stage",
        )
        acost = gp.Variable(
            m,
            "sddp_acost",
            description="total approximate cost: stage_cost + alpha[next stage]",
        )
        self._alpha = alpha
        self._acost = acost

        # Step 5: obj_approx
        # acost == stage_cost + alpha[w+1]
        # For the last stage, Ord(w) < Card(w) is False -> Sum contributes 0.
        obj_approx = gp.Equation(
            m, "sddp_obj_approx", description="approximate objective with future cost"
        )
        obj_approx[...] = acost == stage_cost + gp.Sum(
            w.where[ww[w] & (gp.Ord(w) < gp.Card(w))],
            alpha[w.lead(1)],
        )

        # Step 6: Benders cuts
        # alpha[w+1] - sum_s cont_m_s[jj,i,w+1] * res_s[last_of_w] >= delta[jj,i,w]
        # The slope term SUMS over states (a single supporting hyperplane per
        # cut, with one slope per state); the intercept is the shared `sddp_d`.
        cuts = gp.Equation(
            m,
            "sddp_cuts",
            domain=[j, i_set, w],
            description="Benders cuts on future cost",
        )
        slope_total = None
        for sv_k in self._states:
            assert sv_k.cut_slope is not None
            term = gp.Sum(
                tt.where[last_set[w, tt]],
                sv_k.cut_slope[jj, i_set, w.lead(1)] * sv_k.variable[tt],
            )
            slope_total = term if slope_total is None else slope_total + term
        assert slope_total is not None
        cuts[jj, i_set, w].where[ww[w] & (gp.Ord(w) < gp.Card(w))] = (
            alpha[w.lead(1)] - slope_total >= cut_intercept[jj, i_set, w]
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
            m,
            "sddp_model",
            problem="lp",
            equations=list(equations) + [obj_approx, cuts],
            sense=gp.Sense.MIN,
            objective=acost,
        )

        # Step 8: Backward GUSS dict
        # For each (i, s): fix every state at its trial level, inject inflow[s],
        # then collect acost.l and, PER STATE, x_s.m[prev_last] -> that state's
        # cut slope.
        is_scen = gp.Set(
            m,
            "sddp_is_scen",
            domain=[i_set, s_set],
            records=[(il, sl) for il in i_labels for sl in s_labels],
            description="backward GUSS scenario set (trial x inflow)",
        )
        so = gp.Parameter(
            m,
            "sddp_so",
            domain="*",
            records=[("SkipBaseCase", 1), ("LogOption", 1), ("UpdateType", 2)],
        )
        self._so = so
        is_noise_b = gp.Parameter(
            m,
            "sddp_is_noise_b",
            domain=[i_set, s_set],
            description="scatter: inflow per backward scenario",
        )
        is_acost = gp.Parameter(
            m,
            "sddp_is_acost",
            domain=[i_set, s_set],
            description="extract: acost.l per backward scenario",
        )

        for sv_k in self._states:
            sv_k.is_res_fx = gp.Parameter(
                m,
                f"sddp_is_res_fx_{sv_k.name}",
                domain=[i_set, s_set, t_set],
                description=f"scatter: fix {sv_k.name} at trial level",
            )
            sv_k.is_res_m = gp.Parameter(
                m,
                f"sddp_is_res_m_{sv_k.name}",
                domain=[i_set, s_set, t_set],
                description=f"extract: {sv_k.name}.m[prev_last] = cut slope",
            )
            sv_k.guss_cm = gp.Parameter(
                m,
                f"sddp_gcm_{sv_k.name}",
                domain=i_set,
                description=f"cut slope accumulator for {sv_k.name}",
            )

        dict_b = gp.GUSSScenarioDict(m, "sddp_dict_b", is_scen)
        dict_b.add_options(so)
        dict_b.add_param(nc.parameter, is_noise_b)  # inject scalar inflow
        for sv_k in self._states:
            assert sv_k.is_res_fx is not None
            dict_b.add_fixed(sv_k.variable, sv_k.is_res_fx)  # fix at trial level
        dict_b.add_level(acost, is_acost)  # collect acost.l
        for sv_k in self._states:
            assert sv_k.is_res_m is not None
            dict_b.add_marginal(sv_k.variable, sv_k.is_res_m)  # collect x_s.m

        # Scenario inflow as a GAMSPy parameter for use in symbolic loop assignments
        sw_inflow_param = gp.Parameter(
            m,
            "sddp_sw_inflow",
            domain=[w, s_set],
            records=pd.DataFrame(
                [
                    (w_labels[wi], sl, float(nc.scenario_data[wi, si]))
                    for wi in range(n_stages)
                    for si, sl in enumerate(s_labels)
                ],
                columns=["w", "s", "value"],
            ),
            description="stochastic inflow data (n_stages x n_scenarios)",
        )

        # Cut intercept accumulator (shared: a single intercept per cut).
        guss_d = gp.Parameter(
            m,
            "sddp_gd",
            domain=i_set,
            description="cut intercept accumulator (per trial point)",
        )

        # CVaR change-of-measure weight accumulators
        sp = gp.Alias(m, "sddp_sp", alias_with=s_set)
        cvar_A = gp.Parameter(
            m,
            "sddp_cvar_A",
            domain=[i_set, s_set],
            description="CVaR: prob mass of scenarios worse than s, per trial",
        )
        cvar_w = gp.Parameter(
            m,
            "sddp_cvar_w",
            domain=[i_set, s_set],
            description="CVaR: backward blended change-of-measure weight",
        )

        # Step 9: Forward GUSS dict
        # n_trials scenarios per stage: one sampled inflow path per trial level.
        sampled_path = gp.Set(
            m,
            "sddp_sampled_path",
            domain=[j, w, i_set, s_set],
            description="pre-sampled inflow scenario for each (iter, stage, trial)",
        )  # Path determined in train()

        f_scen = gp.Set(
            m,
            "sddp_f_scen",
            domain=[i_set, s_set],
            description="active forward scenarios for current stage",
        )
        f_inflow = gp.Parameter(
            m,
            "sddp_f_inflow",
            domain=[i_set, s_set],
            description="scatter: inflow per forward scenario",
        )
        f_acost = gp.Parameter(
            m,
            "sddp_f_acost",
            domain=[i_set, s_set],
            description="extract: acost.l per forward scenario",
        )
        f_cost = gp.Parameter(
            m,
            "sddp_f_cost",
            domain=[i_set, s_set],
            description="extract: stage_cost.l (without alpha)",
        )
        for sv_k in self._states:
            sv_k.f_res_fixed = gp.Parameter(
                m,
                f"sddp_f_res_fixed_{sv_k.name}",
                domain=[i_set, s_set, t_set],
                description=f"scatter: fix {sv_k.name} at forward state",
            )
            sv_k.f_res_level = gp.Parameter(
                m,
                f"sddp_f_res_level_{sv_k.name}",
                domain=[i_set, s_set, t_set],
                description=f"extract: {sv_k.name}.l after forward solve",
            )

        dict_f = gp.GUSSScenarioDict(m, "sddp_dict_f", f_scen)
        dict_f.add_options(so)
        dict_f.add_param(nc.parameter, f_inflow)  # inject inflow
        for sv_k in self._states:
            assert sv_k.f_res_fixed is not None
            dict_f.add_fixed(sv_k.variable, sv_k.f_res_fixed)  # fix at forward state
        dict_f.add_level(acost, f_acost)  # collect acost.l (for diagnostics)
        dict_f.add_level(stage_cost, f_cost)  # collect stage cost (for zt)
        for sv_k in self._states:
            assert sv_k.f_res_level is not None
            dict_f.add_level(sv_k.variable, sv_k.f_res_level)  # res.l (next stage)

        # Step 9b: Stage-1 wait-and-see GUSS dict
        # n_scenarios LPs in a single GUSS batch, one per stage-1 realization
        # of the noise.
        is_w1_noise = gp.Parameter(
            m,
            "sddp_is_w1_noise",
            domain=s_set,
            description="scatter: precip per stage-1 scenario",
        )
        is_w1_acost = gp.Parameter(
            m,
            "sddp_is_w1_acost",
            domain=s_set,
            description="extract: acost.l per stage-1 scenario (LB)",
        )
        is_w1_cost = gp.Parameter(
            m,
            "sddp_is_w1_cost",
            domain=s_set,
            description="extract: stage_cost.l per stage-1 scenario",
        )
        for sv_k in self._states:
            sv_k.is_w1_init = gp.Parameter(
                m,
                f"sddp_is_w1_init_{sv_k.name}",
                domain=[s_set, t_set],
                description=f"scatter: fix {sv_k.name} at initial state for stage 1",
            )
            sv_k.is_w1_lstate = gp.Parameter(
                m,
                f"sddp_is_w1_lstate_{sv_k.name}",
                domain=[s_set, t_set],
                description=f"extract: {sv_k.name}.l per stage-1 scenario",
            )

        dict_w1 = gp.GUSSScenarioDict(m, "sddp_dict_w1", s_set)
        dict_w1.add_options(so)
        dict_w1.add_param(nc.parameter, is_w1_noise)
        for sv_k in self._states:
            assert sv_k.is_w1_init is not None
            dict_w1.add_fixed(sv_k.variable, sv_k.is_w1_init)
        dict_w1.add_level(acost, is_w1_acost)
        dict_w1.add_level(stage_cost, is_w1_cost)
        for sv_k in self._states:
            assert sv_k.is_w1_lstate is not None
            dict_w1.add_level(sv_k.variable, sv_k.is_w1_lstate)

        # Step 10: Loop sets and convergence bookkeeping
        wp = gp.Alias(m, "sddp_wp", alias_with=w)
        wloop = gp.Alias(m, "sddp_wloop", alias_with=w)
        wfwd = gp.Alias(m, "sddp_wfwd", alias_with=w)

        # revt[wp, wloop]: wp=w1..w51 paired with wloop=w52..w2 (reversed)
        revt = gp.Set(
            m,
            "sddp_revt",
            domain=[wp, wloop],
            records=[(w_labels[k], w_labels[-(k + 1)]) for k in range(n_stages - 1)],
            description="backward pass index map",
        )
        wfwd_set = gp.Set(
            m,
            "sddp_wfwd_set",
            domain=[wfwd],
            records=w_labels[1:],
            description="forward pass stages (w2..last)",
        )

        # Convergence and state-tracking parameters
        conv = gp.Parameter(
            m,
            "sddp_conv",
            domain=[j, "*"],
            description="convergence bounds per iteration",
        )
        zt = gp.Parameter(
            m,
            "sddp_zt",
            domain=[j, i_set],
            description="accumulated stage cost per forward path",
        )
        for sv_k in self._states:
            sv_k.forward_state = gp.Parameter(
                m,
                f"sddp_fstate_{sv_k.name}",
                domain=i_set,
                description=f"{sv_k.name} state at end of current forward stage",
            )
            sv_k.forward_res_state = gp.Parameter(
                m,
                f"sddp_frstate_{sv_k.name}",
                domain=[j, w, i_set],
                description=f"{sv_k.name} state history for adaptive trials",
            )

        # Adaptive-update support symbols
        adapt = gp.Set(
            m,
            "sddp_adapt",
            domain=[t_set, w],
            records=[(prev_last_hour[wl], wl) for wl in w_labels[1:]],
            description="adaptive trial update: (prev_last_hour, downstream stage)",
        )
        adapt_t = gp.Set(
            m,
            "sddp_adapt_t",
            domain=t_set,
            records=[prev_last_hour[wl] for wl in w_labels[1:]],
            description="t indices the adaptive trial update writes to",
        )
        prev_j_indicator = gp.Parameter(
            m,
            "sddp_prev_j",
            domain=j,
            description="indicator (0/1) selecting the previous iteration for adaptive update",
        )

        # Store everything train() will need
        self._w = w
        self._w_labels = w_labels
        self._t_labels = t_labels
        self._last_hour = last_hour
        self._prev_last_hour = prev_last_hour
        self._hpw = hpw
        self._last_set = last_set
        self._prevlast_set = prevlast_set
        self._j_labels: list[str] = []  # populated by train() once n_iter is set
        self._s_labels = s_labels
        self._i_labels = i_labels
        self._stage_cost_var = stage_cost
        self._user_equations = list(equations)
        self._obj_approx_eq = obj_approx
        self._cuts_eq = cuts
        self._solve_opts = solve_opts
        self._gp_model = gp_model
        self._is_scen = is_scen
        self._is_noise_b = is_noise_b
        self._is_acost = is_acost
        self._dict_b = dict_b
        self._sw_inflow_param = sw_inflow_param
        self._guss_d = guss_d
        self._sp = sp
        self._cvar_A = cvar_A
        self._cvar_w = cvar_w
        self._sampled_path = sampled_path
        self._f_scen = f_scen
        self._f_inflow = f_inflow
        self._f_acost = f_acost
        self._f_cost = f_cost
        self._dict_f = dict_f
        self._dict_w1 = dict_w1
        self._is_w1_noise = is_w1_noise
        self._is_w1_acost = is_w1_acost
        self._is_w1_cost = is_w1_cost
        self._wp = wp
        self._wloop = wloop
        self._wfwd = wfwd
        self._revt = revt
        self._wfwd_set = wfwd_set
        self._conv = conv
        self._zt = zt
        self._adapt = adapt
        self._adapt_t = adapt_t
        self._prev_j_ind = prev_j_indicator
        self._tt = tt

        self._state_hour = prev_last_hour[w_labels[0]]
        self._state_orig_lo, self._state_orig_up = self._read_var_bounds(
            sv.variable, self._state_hour
        )

        # Snapshot each state's full user-set bound profile
        type_defaults = {
            "positive": (0.0, float("inf")),
            "negative": (float("-inf"), 0.0),
            "binary": (0.0, 1.0),
            "integer": (0.0, float("inf")),
        }
        for sv_k in self._states:
            d_lo, d_up = type_defaults.get(
                getattr(sv_k.variable, "type", "free"), (float("-inf"), float("inf"))
            )
            recs = sv_k.variable.records
            lo_rows: list[tuple] = []
            up_rows: list[tuple] = []
            for tl in t_labels:
                lo, up = d_lo, d_up
                if recs is not None and len(recs) > 0 and "lower" in recs.columns:
                    domain_col = recs.columns[0]
                    match = recs[recs[domain_col].astype(str) == tl]
                    if len(match) > 0:
                        lo = float(match["lower"].iloc[0])
                        up = float(match["upper"].iloc[0])
                lo_rows.append((tl, lo))
                up_rows.append((tl, up))

            sv_k.orig_lo_param = gp.Parameter(
                m,
                f"sddp_orig_lo_{sv_k.name}",
                domain=t_set,
                records=pd.DataFrame(lo_rows, columns=["t", "value"]),
                description=f"snapshot of user lower bounds for {sv_k.name}",
            )
            sv_k.orig_up_param = gp.Parameter(
                m,
                f"sddp_orig_up_{sv_k.name}",
                domain=t_set,
                records=pd.DataFrame(up_rows, columns=["t", "value"]),
                description=f"snapshot of user upper bounds for {sv_k.name}",
            )

        value_cols = {"level", "marginal", "lower", "upper", "scale"}
        state_names = {s.name for s in self._states}
        self._user_bound_snaps: list[tuple] = []
        for v in self._user_variables:
            if v.name in state_names:
                continue
            vrecs = v.records
            if vrecs is None or len(vrecs) == 0:
                continue
            if (vrecs["lower"] == float("-inf")).all() and (
                vrecs["upper"] == float("inf")
            ).all():
                continue  # fully free variable
            vdom = list(v.domain)
            dom_cols = [c for c in vrecs.columns if c not in value_cols]
            lo_p = gp.Parameter(
                m,
                f"sddp_blo_{v.name}",
                domain=vdom,
                records=vrecs[[*dom_cols, "lower"]].rename(columns={"lower": "value"}),
                description=f"snapshot of user lower bounds for {v.name}",
            )
            up_p = gp.Parameter(
                m,
                f"sddp_bup_{v.name}",
                domain=vdom,
                records=vrecs[[*dom_cols, "upper"]].rename(columns={"upper": "value"}),
                description=f"snapshot of user upper bounds for {v.name}",
            )
            self._user_bound_snaps.append((v, vdom, lo_p, up_p))

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
    def _read_var_bounds(var: gp.Variable, idx: str) -> tuple[float, float]:
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
            return default_lo, default_up
        if "lower" not in recs.columns or "upper" not in recs.columns:
            return default_lo, default_up

        domain_col = recs.columns[0]
        match = recs[recs[domain_col].astype(str) == idx]
        if len(match) == 0:
            return default_lo, default_up
        return float(match["lower"].iloc[0]), float(match["upper"].iloc[0])

    @staticmethod
    def _cvar_backward_weights(
        risk: CVaR,
        cvar_A: gp.Parameter,
        cvar_w: gp.Parameter,
        cost: gp.Parameter,
        prob_param: gp.Parameter,
        sp: gp.Alias,
        i_set: gp.Set,
        s_set: gp.Set,
    ) -> None:

        cvar_A[i_set, s_set] = gp.Sum(
            sp.where[
                (cost[i_set, sp] > cost[i_set, s_set])
                | (
                    (cost[i_set, sp] >= cost[i_set, s_set])
                    & (gp.Ord(sp) < gp.Ord(s_set))
                )
            ],
            prob_param[sp],
        )
        cvar_w[i_set, s_set] = (1.0 - risk.weight) * prob_param[s_set] + (
            risk.weight
            * gp.math.Max(
                0.0,
                gp.math.Min(prob_param[s_set], risk.tail - cvar_A[i_set, s_set]),
            )
            / risk.tail
        )

    def _set_active_cuts(self, completed: int, keep_iter: int | None) -> None:
        """Point the active-cut set at a window of the completed iterations.

        Cuts are only ever deactivated, never deleted: ``cut_slope`` and
        ``cut_intercept`` keep every value they were given, and ``jj`` decides
        which of them become LP rows. Restoring the whole pool is therefore a
        single set assignment rather than a recomputation.

        Parameters
        ----------
        completed : int
            Number of iterations finished so far, i.e. labels ``j1..j{completed}``.
        keep_iter : int | None
            Retain only the most recent ``keep_iter`` of them; ``None`` retains
            all of them.
        """
        assert self._j_set is not None
        assert self._jj_set is not None
        j = self._j_set
        jj = self._jj_set

        first = 1 if keep_iter is None else max(1, completed - keep_iter + 1)
        jj[j] = False
        jj[j].where[(gp.Ord(j) >= first) & (gp.Ord(j) <= completed)] = True

    def _stage1_bound(self, slot: str) -> float:
        """Expected stage-1 cost under whichever cuts are currently active.

        This is what the lower bound *is*, so pricing a different cut pool needs
        only this one wait-and-see batch: the pool is being evaluated, not
        extended, so no backward pass is involved. Costs ``n_scenarios`` LPs
        against the several thousand a full iteration solves.
        """
        assert self._noise is not None
        assert self._gp_model is not None
        s_set = self._noise.scenario_set
        assert s_set is not None

        m = self._m
        w = self._w
        ww = self._active_stage
        t_set = self._time_set
        conv = self._conv
        w1 = self._w_labels[0]

        # Same single-sync pattern as the training loop: batch every statement
        # and this solve into one GAMS job rather than a round-trip apiece.
        m._in_loop += 1
        try:
            ww[w] = False
            ww[w1] = True
            for sv_k in self._states:
                assert sv_k.is_w1_init is not None
                assert sv_k.is_w1_lstate is not None
                assert sv_k.orig_lo_param is not None
                assert sv_k.orig_up_param is not None
                sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
                sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]
                sv_k.is_w1_init[s_set, t_set] = 0
                sv_k.is_w1_init[s_set, self._state_hour] = max(
                    sv_k.initial_state
                    if sv_k.initial_state is not None
                    else sv_k.lower_bound,
                    self._EPS,
                )
                sv_k.is_w1_lstate[s_set, t_set] = 0

            self._is_w1_noise[s_set] = self._sw_inflow_param[w1, s_set]
            self._is_w1_acost[s_set] = 0
            self._is_w1_cost[s_set] = 0

            self._gp_model.solve(options=self._solve_opts, scenario=self._dict_w1)

            conv[slot, "lo_full"] = gp.Sum(
                s_set, self._prob_param[s_set] * self._is_w1_acost[s_set]
            )
        finally:
            m._in_loop -= 1
        m._synch_with_gams()

        return float(_pv_at_j(conv, slot).get("lo_full", 0.0))

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
        assert self._j_set is not None
        assert self._jj_set is not None
        assert self._alpha is not None
        assert self._acost is not None
        assert self._gp_model is not None

        m = self._m
        ww = self._active_stage
        w = self._w
        t = self._time_set
        t_set = t

        sv = self._states[0]
        assert sv.trial_set is not None
        assert sv.trial_param is not None
        assert sv.cut_slope is not None

        nc = self._noise
        assert nc.scenario_set is not None
        i_set = sv.trial_set
        s_set = nc.scenario_set
        j = self._j_set
        jj = self._jj_set

        w_labels = self._w_labels
        i_labels = self._i_labels
        s_labels = self._s_labels

        last_set = self._last_set
        prevlast_set = self._prevlast_set

        gp_model = self._gp_model
        solve_opts = self._solve_opts
        dict_b = self._dict_b
        dict_f = self._dict_f
        prob_param = self._prob_param

        is_noise_b = self._is_noise_b
        is_acost = self._is_acost
        guss_d = self._guss_d
        cut_intercept = self._cut_intercept
        sp = self._sp
        cvar_A = self._cvar_A
        cvar_w = self._cvar_w

        f_scen = self._f_scen
        f_inflow = self._f_inflow
        f_acost = self._f_acost
        f_cost = self._f_cost

        dict_w1 = self._dict_w1
        is_w1_noise = self._is_w1_noise
        is_w1_acost = self._is_w1_acost
        is_w1_cost = self._is_w1_cost

        wp = self._wp
        wloop = self._wloop
        wfwd = self._wfwd
        revt = self._revt
        wfwd_set = self._wfwd_set
        sampled_path = self._sampled_path
        sw_inflow = self._sw_inflow_param

        conv = self._conv
        zt = self._zt
        adapt = self._adapt
        adapt_t = self._adapt_t
        prev_j_ind = self._prev_j_ind
        tt = self._tt

        state_hour = self._state_hour

        # Size the iteration machinery for this run
        j.setRecords([f"j{k}" for k in range(1, n_iter + 1)])
        j_labels = j.toList()
        self._j_labels = j_labels

        rng = np.random.default_rng(seed=self._seed)
        sample_p = nc.probabilities  # None -> uniform; preserves seed sequences
        sampled_path.setRecords(
            [
                (jl, wl, il, rng.choice(s_labels, p=sample_p))
                for jl in j_labels
                for wl in w_labels
                for il in i_labels
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

        for pos, j_label in enumerate(j_labels):
            # A CTRL+C during the previous iteration's tail -> stop cleanly here.
            if self._stop_requested:
                result.stop_reason = "interrupted"
                break

            prev_j = j_labels[pos - 1] if pos > 0 else None
            t_iter = time.perf_counter()
            interrupted = False

            # Single-sync pattern: pre-bump prevents nested Loop.__exit__ syncs.
            # One explicit _synch_with_gams() at the end covers the full iteration.
            m._in_loop += 1
            try:
                ##########          Adaptive trial-level update          ##########
                # Replace the static uniform grid with the previous iteration's
                # forward-pass reservoir states. The whole update is a single
                # symbolic GAMS assignment; the `prev_j_ind` parameter selects
                # which iteration's frs to pull from, and the `adapt` set
                # threads (prev_last_hour[wl], wl) so the inner Sum has exactly
                # one non-zero term per (i, t). Clamp + eps floor are done
                # in-line via gp.math.Max/Min so we still get only one GAMS
                # statement (vs. ~n_trials x (n_stages - 1) per-iter before).
                if prev_j is not None:
                    prev_j_ind[j] = 0
                    prev_j_ind[prev_j] = 1
                    for sv_k in self._states:
                        assert sv_k.trial_param is not None
                        assert sv_k.forward_res_state is not None
                        sv_k.trial_param[i_set, tt].where[adapt_t[tt]] = gp.math.Max(
                            self._EPS,
                            gp.math.Min(
                                sv_k.upper_bound,
                                gp.math.Max(
                                    sv_k.lower_bound,
                                    gp.Sum(
                                        gp.Domain(j, w).where[
                                            prev_j_ind[j] & adapt[tt, w]
                                        ],
                                        sv_k.forward_res_state[j, w, i_set],
                                    ),
                                ),
                            ),
                        )

                # Reset per-iteration accumulators
                zt[j_label, i_set] = 0
                for sv_k in self._states:
                    assert sv_k.forward_state is not None
                    assert sv_k.forward_res_state is not None
                    sv_k.forward_state[i_set] = 0
                    sv_k.forward_res_state[j_label, w, i_set] = 0

                # Backward pass
                with gp.Loop(gp.Domain(wp, wloop).where[revt[wp, wloop]]):
                    # Activate the current backward stage
                    ww[w] = False
                    ww[wloop] = True

                    # Restore user bounds + scatter each state's trial level.
                    # GUSS leaves a state variable fixed at the prior batch's
                    # last scenario, which would pin the current stage's free
                    # decision variable at a stale value.
                    for sv_k in self._states:
                        assert sv_k.is_res_fx is not None
                        assert sv_k.trial_param is not None
                        assert sv_k.orig_lo_param is not None
                        assert sv_k.orig_up_param is not None
                        sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
                        sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]
                        # Scatter: fix x_s[prev_last_of_wloop] at trial level[i]
                        sv_k.is_res_fx[i_set, s_set, t_set] = 0
                        sv_k.is_res_fx[i_set, s_set, tt].where[
                            prevlast_set[wloop, tt]
                        ] = sv_k.trial_param[i_set, tt]

                    # Scatter: inject inflow scenario for this stage (same for all i)
                    is_noise_b[i_set, s_set] = sw_inflow[wloop, s_set]

                    # GUSS solve: (n_trials x n_scenarios) LPs in one batch
                    gp_model.solve(options=solve_opts, scenario=dict_b)

                    # Change-of-measure weight per scenario: nominal probability,
                    # or the CVaR Rockafellar-Uryasev weight when risk-averse.
                    if isinstance(risk, CVaR):
                        self._cvar_backward_weights(
                            risk, cvar_A, cvar_w, is_acost, prob_param, sp, i_set, s_set
                        )
                        weight = cvar_w[i_set, s_set]
                    else:
                        weight = prob_param[s_set]

                    # Cut slope, per state: weighted x_s.m[prev_last] over scenarios.
                    for sv_k in self._states:
                        assert sv_k.guss_cm is not None
                        assert sv_k.is_res_m is not None
                        sv_k.guss_cm[i_set] = gp.Sum(
                            gp.Domain(s_set, tt).where[prevlast_set[wloop, tt]],
                            weight * sv_k.is_res_m[i_set, s_set, tt],
                        )

                    # Cut intercept (shared): V(trial) - sum_s slope_s * trial_s.
                    # Uses LP strong duality: delta = E[acost.l] - sum_s slope_s*x0_s.
                    slope_dot_trial = None
                    for sv_k in self._states:
                        assert sv_k.guss_cm is not None
                        assert sv_k.trial_param is not None
                        term = sv_k.guss_cm[i_set] * gp.Sum(
                            tt.where[prevlast_set[wloop, tt]],
                            sv_k.trial_param[i_set, tt],
                        )
                        slope_dot_trial = (
                            term if slope_dot_trial is None else slope_dot_trial + term
                        )
                    assert slope_dot_trial is not None
                    guss_d[i_set] = (
                        gp.Sum(s_set, weight * is_acost[i_set, s_set]) - slope_dot_trial
                    )

                    # Store cut: per-state slope, shared intercept; activate jj.
                    for sv_k in self._states:
                        assert sv_k.cut_slope is not None
                        assert sv_k.guss_cm is not None
                        sv_k.cut_slope[j_label, i_set, wloop] = sv_k.guss_cm[i_set]
                    cut_intercept[j_label, i_set, wloop.lag(1)] = guss_d[i_set]
                    jj[j_label] = True

                # Forward stage 1: wait-and-see GUSS batch
                ww[w] = False
                ww[w_labels[0]] = True

                # Restore user bounds + fix each state at its initial level.
                # GUSS leaves variables at the prior batch's last scenario, same
                # as the backward sweep.
                for sv_k in self._states:
                    assert sv_k.is_w1_init is not None
                    assert sv_k.is_w1_lstate is not None
                    assert sv_k.orig_lo_param is not None
                    assert sv_k.orig_up_param is not None
                    sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
                    sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]
                    sv_k.is_w1_init[s_set, t_set] = 0
                    sv_k.is_w1_init[s_set, state_hour] = max(
                        sv_k.initial_state
                        if sv_k.initial_state is not None
                        else sv_k.lower_bound,
                        self._EPS,
                    )
                    sv_k.is_w1_lstate[s_set, t_set] = 0

                # Scatter: precip per scenario (shared)
                is_w1_noise[s_set] = sw_inflow[w_labels[0], s_set]
                is_w1_acost[s_set] = 0
                is_w1_cost[s_set] = 0

                gp_model.solve(options=solve_opts, scenario=dict_w1)

                conv[j_label, "lo"] = gp.Sum(
                    s_set, prob_param[s_set] * is_w1_acost[s_set]
                )

                # Per-trial stage-1 cost: pick each trial's pre-sampled scenario
                zt[j_label, i_set] = gp.Sum(
                    s_set.where[sampled_path[j_label, w_labels[0], i_set, s_set]],
                    is_w1_cost[s_set],
                )

                # Per-trial forward state per state: x_s[last_of_w1] from the
                # trial's sampled scenario.
                for sv_k in self._states:
                    assert sv_k.forward_state is not None
                    assert sv_k.forward_res_state is not None
                    assert sv_k.is_w1_lstate is not None
                    sv_k.forward_state[i_set] = gp.Sum(
                        gp.Domain(s_set, tt).where[
                            sampled_path[j_label, w_labels[0], i_set, s_set]
                            & last_set[w_labels[0], tt]
                        ],
                        sv_k.is_w1_lstate[s_set, tt],
                    )
                    sv_k.forward_res_state[j_label, w_labels[1], i_set] = (
                        sv_k.forward_state[i_set]
                    )

                # Forward pass:
                with gp.Loop(wfwd.where[wfwd_set[wfwd]]):
                    ww[w] = False
                    ww[wfwd] = True

                    # Select the pre-sampled inflow scenario for each trial point
                    f_scen[i_set, s_set] = False
                    f_scen[i_set, s_set].where[
                        sampled_path[j_label, wfwd, i_set, s_set]
                    ] = True

                    # Scatter: inject inflow from sampled scenario
                    f_inflow[i_set, s_set] = 0
                    f_inflow[i_set, s_set].where[f_scen[i_set, s_set]] = sw_inflow[
                        wfwd, s_set
                    ]

                    # Per state: restore bounds, fix x_s[prev_last] at the forward
                    # state, reset its level extract.
                    for sv_k in self._states:
                        assert sv_k.f_res_fixed is not None
                        assert sv_k.f_res_level is not None
                        assert sv_k.forward_state is not None
                        assert sv_k.orig_lo_param is not None
                        assert sv_k.orig_up_param is not None
                        sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
                        sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]
                        sv_k.f_res_fixed[i_set, s_set, t_set] = 0
                        sv_k.f_res_fixed[i_set, s_set, tt].where[
                            f_scen[i_set, s_set] & prevlast_set[wfwd, tt]
                        ] = sv_k.forward_state[i_set]
                        sv_k.f_res_level[i_set, s_set, t_set] = 0

                    # Reset shared extract containers
                    f_acost[i_set, s_set] = 0
                    f_cost[i_set, s_set] = 0

                    # GUSS solve
                    gp_model.solve(options=solve_opts, scenario=dict_f)

                    # Accumulate stage cost into zt
                    zt[j_label, i_set] = zt[j_label, i_set] + gp.Sum(
                        s_set.where[f_scen[i_set, s_set]],
                        f_cost[i_set, s_set],
                    )

                    # Advance each state for the next stage
                    for sv_k in self._states:
                        assert sv_k.f_res_level is not None
                        assert sv_k.forward_state is not None
                        assert sv_k.forward_res_state is not None
                        sv_k.forward_state[i_set] = gp.Sum(
                            gp.Domain(s_set, tt).where[
                                f_scen[i_set, s_set] & last_set[wfwd, tt]
                            ],
                            sv_k.f_res_level[i_set, s_set, tt],
                        )
                        sv_k.forward_res_state[j_label, wfwd.lead(1), i_set] = (
                            sv_k.forward_state[i_set]
                        )

                # Upper bound
                # Average total-path cost across all n_trials forward trajectories
                conv[j_label, "up"] = gp.Sum(i_set, zt[j_label, i_set]) / gp.Card(i_set)

                # Cut selection: retire the iteration that has just fallen out
                # of the window.
                if cut_selection is not None:
                    expire = pos - cut_selection.keep_iter
                    if expire >= 0:
                        jj[j_labels[expire]] = False
            except GamspyException:
                if self._stop_requested:
                    result.stop_reason = "interrupted"
                    interrupted = True
                else:
                    _restore_sigint()
                    raise
            finally:
                m._in_loop -= 1

            # Single explicit sync, now that the pre-bump is released so it runs
            # at the baseline _in_loop and actually flushes. Wrapped so a CTRL+C
            # that aborts the sync's own GAMS job is finalized too.
            if not interrupted:
                try:
                    m._synch_with_gams()
                except GamspyException:
                    if self._stop_requested:
                        result.stop_reason = "interrupted"
                        interrupted = True
                    else:
                        _restore_sigint()
                        raise

            if interrupted:
                break

            # Safety: jj must round-trip back to Python after every iteration.
            # If it doesn't, downstream policy()/simulate() solves see jj empty,
            # the Benders cuts deactivate, alpha collapses to its zero bound,
            # and decisions are silently wrong. Catches breakage of the
            # end-of-iteration sync at the moment it happens, not three steps
            # downstream when a user sees a nonsense policy result.
            active_labels = 0 if jj.records is None else len(jj.records)
            expected_labels = (
                min(pos + 1, cut_selection.keep_iter)
                if cut_selection is not None
                else pos + 1
            )
            if active_labels != expected_labels:
                _restore_sigint()
                raise GamspyException(
                    f"jj out of step after iteration {j_label}: expected "
                    f"{expected_labels} active cut iteration(s), got "
                    f"{active_labels}. Either the end-of-iteration sync did not "
                    f"bring `jj` back to Python, or cut selection retired the wrong "
                    f"label."
                )

            active_cuts = active_labels * self._n_trials

            # Read back convergence bounds
            # Filter for this iteration's rows
            cv = _pv_at_j(conv, j_label)
            lo = cv.get("lo", 0.0)
            elapsed = time.perf_counter() - t_iter

            # Forward-pass cost estimate from this iteration's trial paths
            # zt[j_label, il] = total cost on path il.
            zt_row = _pv_at_j(zt, j_label)
            path_costs = np.array([zt_row.get(il, 0.0) for il in i_labels])
            n_paths = len(path_costs)
            up = float(np.mean(path_costs))
            sigma = float(np.std(path_costs, ddof=1)) if n_paths > 1 else 0.0
            up_95 = up + 1.96 * sigma / np.sqrt(n_paths)

            # Cut selection makes the bound non-monotone by design: retiring a
            # binding cut may lower it.
            if cut_selection is None and prev_lo is not None and lo < prev_lo - 1e-3:
                _restore_sigint()
                raise GamspyException(
                    f"Lower bound decreased at {j_label}: {prev_lo:,.3f} -> {lo:,.3f}"
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
                "iteration": j_label,
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
                    f"  {j_label:>4s}:  "
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
                    full_lo = self._stage1_bound(j_label)
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
                                f"  {j_label:>4s}:  plateau overruled - full "
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
                lo = self._stage1_bound(j_labels[completed - 1])
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
        _pack(self._m, metadata, path)

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
        for v in resolved:
            if list(v.domain) != [self._time_set]:
                raise NotImplementedError(
                    f"simulate() v1 only captures variables with domain==[time_set]; "
                    f"got {v.name} with domain={v.domain}"
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
        t0 = time.perf_counter()

        # build() invariants: narrow Optional fields for the type checker.
        assert self._noise is not None
        assert self._jj_set is not None
        assert self._gp_model is not None
        assert self._acost is not None

        m = self._m
        sv = self._states[0]
        assert sv.trial_set is not None
        assert sv.cut_slope is not None
        assert sv.orig_lo_param is not None
        assert sv.orig_up_param is not None
        nc = self._noise
        assert nc.scenario_set is not None
        s_set = nc.scenario_set
        w = self._w
        t_set = self._time_set
        tt = self._tt
        ww = self._active_stage
        w_labels = self._w_labels
        s_labels = self._s_labels
        last_set = self._last_set
        prevlast_set = self._prevlast_set
        state_hour = self._state_hour
        sw_inflow = self._sw_inflow_param

        # Every state variable's post-solve level drives forward propagation, so
        # it must be extracted even if the caller's `report` omits it.
        state_vars = [sv_k.variable for sv_k in self._states]
        _report_names = {v.name for v in report}
        extract_vars = list(report) + [
            v for v in state_vars if v.name not in _report_names
        ]

        sfx = f"_{call_id}"

        # Build per-call simulation symbols
        p_labels = [f"p{k}" for k in range(1, n_paths + 1)]
        sim_p = gp.Set(
            m,
            f"sddp_sim_p{sfx}",
            records=p_labels,
            description="simulation paths",
        )

        rng = np.random.default_rng(seed)
        sample_p = nc.probabilities  # None -> uniform; preserves seed sequences
        sample_recs = [
            (p, wl, rng.choice(s_labels, p=sample_p))
            for p in p_labels
            for wl in w_labels
        ]
        sim_sample = gp.Set(
            m,
            f"sddp_sim_sample{sfx}",
            domain=[sim_p, w, s_set],
            records=sample_recs,
            description="pre-sampled noise scenario per (path, stage)",
        )

        # Active (path, scenario) tuples for the current stage, mutated in loop.
        sim_active = gp.Set(
            m,
            f"sddp_sim_active{sfx}",
            domain=[sim_p, s_set],
            description="active simulation scenarios for current stage",
        )

        # Scatter parameters
        sim_noise = gp.Parameter(
            m,
            f"sddp_sim_noise{sfx}",
            domain=[sim_p, s_set],
            description="scatter: noise per simulation scenario",
        )
        sim_state: dict[str, gp.Parameter] = {}
        for sv_k in self._states:
            sim_state[sv_k.variable.name] = gp.Parameter(
                m,
                f"sddp_sim_state_{sv_k.name}{sfx}",
                domain=[sim_p, s_set, t_set],
                description=f"scatter: fix {sv_k.name} per simulation scenario",
            )

        # Extract parameters
        sim_acost = gp.Parameter(
            m,
            f"sddp_sim_acost{sfx}",
            domain=[sim_p, s_set],
            description="extract: acost.l per simulation scenario",
        )
        sim_cost = gp.Parameter(
            m,
            f"sddp_sim_cost{sfx}",
            domain=[sim_p, s_set],
            description="extract: stage_cost.l per simulation scenario",
        )
        sim_var_extract: dict[str, gp.Parameter] = {}
        for v in extract_vars:
            sim_var_extract[v.name] = gp.Parameter(
                m,
                f"sddp_sim_var_{v.name}{sfx}",
                domain=[sim_p, s_set, t_set],
                description=f"extract: {v.name}.l per simulation scenario",
            )

        # Per-stage history (path x stage)
        sim_cost_hist = gp.Parameter(
            m,
            f"sddp_sim_cost_hist{sfx}",
            domain=[sim_p, w],
            description="stage cost per (path, stage)",
        )
        sim_noise_hist = gp.Parameter(
            m,
            f"sddp_sim_noise_hist{sfx}",
            domain=[sim_p, w],
            description="realised noise per (path, stage)",
        )
        sim_var_hist: dict[str, gp.Parameter] = {}
        for v in report:
            sim_var_hist[v.name] = gp.Parameter(
                m,
                f"sddp_sim_{v.name}_hist{sfx}",
                domain=[sim_p, w],
                description=f"history: {v.name}.l at last_of_stage per (path, stage)",
            )

        # Forward state tracker per state (path -> end-of-previous-stage level)
        sim_fwd_state: dict[str, gp.Parameter] = {}
        for sv_k in self._states:
            sim_fwd_state[sv_k.variable.name] = gp.Parameter(
                m,
                f"sddp_sim_fwd_state_{sv_k.name}{sfx}",
                domain=sim_p,
                description=f"end-of-previous-stage {sv_k.name} per path",
            )

        # GUSS dict for the simulation
        sim_dict = gp.GUSSScenarioDict(m, f"sddp_sim_dict{sfx}", sim_active)
        sim_dict.add_options(self._so)
        sim_dict.add_param(nc.parameter, sim_noise)
        for sv_k in self._states:
            sim_dict.add_fixed(sv_k.variable, sim_state[sv_k.variable.name])
        sim_dict.add_level(self._acost, sim_acost)
        sim_dict.add_level(self._stage_cost_var, sim_cost)
        for v in extract_vars:
            sim_dict.add_level(v, sim_var_extract[v.name])

        # Forward simulation: one GUSS batch per stage
        m._in_loop += 1
        try:
            for stage_idx, wl in enumerate(w_labels):
                # Activate this stage
                ww[w] = False
                ww[wl] = True

                # Build active (path, scenario) set for this stage
                sim_active[sim_p, s_set] = sim_sample[sim_p, wl, s_set]

                # Scatter noise: only the sampled scenario per path is non-zero
                sim_noise[sim_p, s_set] = 0
                sim_noise[sim_p, s_set].where[sim_active[sim_p, s_set]] = sw_inflow[
                    wl, s_set
                ]

                # Per state: restore bounds and fix the incoming state.
                for sv_k in self._states:
                    assert sv_k.orig_lo_param is not None
                    assert sv_k.orig_up_param is not None
                    sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
                    sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]

                    state_param = sim_state[sv_k.variable.name]
                    state_param[sim_p, s_set, t_set] = 0
                    if stage_idx == 0:
                        initial = (
                            sv_k.initial_state
                            if sv_k.initial_state is not None
                            else sv_k.lower_bound
                        )
                        state_param[sim_p, s_set, state_hour].where[
                            sim_active[sim_p, s_set]
                        ] = max(initial, self._EPS)
                    else:
                        state_param[sim_p, s_set, tt].where[
                            sim_active[sim_p, s_set] & prevlast_set[wl, tt]
                        ] = sim_fwd_state[sv_k.variable.name][sim_p]

                # Reset extracts
                sim_acost[sim_p, s_set] = 0
                sim_cost[sim_p, s_set] = 0
                for v in extract_vars:
                    sim_var_extract[v.name][sim_p, s_set, t_set] = 0

                # GUSS solve: n_paths LPs in one batch
                self._gp_model.solve(options=self._solve_opts, scenario=sim_dict)

                # Record per-stage history
                sim_cost_hist[sim_p, wl] = gp.Sum(
                    s_set.where[sim_active[sim_p, s_set]],
                    sim_cost[sim_p, s_set],
                )
                sim_noise_hist[sim_p, wl] = gp.Sum(
                    s_set.where[sim_active[sim_p, s_set]],
                    sim_noise[sim_p, s_set],
                )
                for v in report:
                    sim_var_hist[v.name][sim_p, wl] = gp.Sum(
                        gp.Domain(s_set, tt).where[
                            sim_active[sim_p, s_set] & last_set[wl, tt]
                        ],
                        sim_var_extract[v.name][sim_p, s_set, tt],
                    )

                # Advance each state's forward value for the next stage.
                if stage_idx < len(w_labels) - 1:
                    for sv_k in self._states:
                        sim_fwd_state[sv_k.variable.name][sim_p] = gp.Sum(
                            gp.Domain(s_set, tt).where[
                                sim_active[sim_p, s_set] & last_set[wl, tt]
                            ],
                            sim_var_extract[sv_k.variable.name][sim_p, s_set, tt],
                        )

        except Exception:
            m._in_loop -= 1
            raise
        m._in_loop -= 1

        # Single sync flushes the loop. The per-stage history params assigned
        # above (sim_cost_hist, sim_noise_hist, sim_*_hist) flag themselves for
        # load-back, so the container loads them automatically.
        m._synch_with_gams()

        # Aggregate into DataFrames
        cost_df = self._pivot_history(sim_cost_hist, p_labels, w_labels)
        noise_df = self._pivot_history(sim_noise_hist, p_labels, w_labels)
        var_dfs = {
            v.name: self._pivot_history(sim_var_hist[v.name], p_labels, w_labels)
            for v in report
        }
        total_cost = cost_df.sum(axis=1)
        total_cost.name = "total_cost"

        elapsed = time.perf_counter() - t0
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
        assert self._jj_set is not None
        assert self._gp_model is not None
        assert self._acost is not None

        if stage not in self._w_labels:
            raise ValidationError(
                f"stage {stage!r} is not a valid stage; "
                f"expected one of {self._w_labels}"
            )

        if self._jj_set.records is None or len(self._jj_set.records) == 0:
            warnings.warn(
                "policy() called before train(); no cuts exist yet, so the "
                "future-cost approximation is absent and the returned decision "
                "is NOT the trained policy. Call train() first.",
                UserWarning,
                stacklevel=2,
            )

        nc = self._noise
        state_values = self._resolve_state(state)

        resolved = self._resolve_report(report)
        for v in resolved:
            if not v.domain or v.domain[0] != self._time_set:
                raise NotImplementedError(
                    f"policy() requires variables whose first domain element "
                    f"is the time set; got {v.name} with "
                    f"domain={v.domain}"
                )

        ww = self._active_stage
        w = self._w
        t_set = self._time_set
        ph = self._prev_last_hour[stage]
        lt = self._last_hour[stage]

        # Activate only this stage
        ww[w] = False
        ww[stage] = True

        # Restore user bounds, then fix each state at its incoming value.
        self._restore_user_bounds()
        for sv_k in self._states:
            assert sv_k.orig_lo_param is not None
            assert sv_k.orig_up_param is not None
            sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
            sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]
            sv_k.variable.fx[ph] = max(state_values[sv_k.variable.name], self._EPS)

        # Inject the realised noise
        nc.parameter[...] = float(noise)

        # Single plain solve
        self._gp_model.solve(options=self._solve_opts)

        cost_to_go = self._scalar_level(self._acost)
        decisions = {v.name: self._extract_decision(v, lt) for v in resolved}

        # Logical cleanup: unfix every state
        for sv_k in self._states:
            assert sv_k.orig_lo_param is not None
            assert sv_k.orig_up_param is not None
            sv_k.variable.lo[t_set] = sv_k.orig_lo_param[t_set]
            sv_k.variable.up[t_set] = sv_k.orig_up_param[t_set]

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
        full_w = getattr(self, "_w", self._stage_parent)
        recs = full_w.records
        return len(recs) if recs is not None else 0

    def __repr__(self) -> str:
        states = [sv.name for sv in self._states]
        noise = self._noise.parameter.name if self._noise else "none"
        return (
            f"SDDP(stages={self.n_stages}, "
            f"states={states}, noise={noise}, built={self._built})"
        )


def _pv_at_j(param: gp.Parameter, j_label: str) -> dict:
    rec = param.records
    if rec is None or len(rec) == 0:
        return {}

    cols = [c for c in rec.columns if c != "value"]
    if not cols:
        return {}

    mask = rec[cols[0]].astype(str).values == j_label
    if not mask.any():
        return {}
    sub = rec.loc[mask]

    other = cols[1:]
    vals = sub["value"].astype(float).values

    if len(other) == 0:
        # Leading column was the only domain, so return a single-entry dict.
        return {(): float(vals[0])}
    if len(other) == 1:
        keys = sub[other[0]].astype(str).values
        return dict(zip(keys, vals, strict=False))
    key_cols = [sub[c].astype(str).values for c in other]
    return dict(zip(zip(*key_cols, strict=False), vals, strict=False))
