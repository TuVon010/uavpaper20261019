"""Independent convex/KKT checks for the time-only association layer."""
from dataclasses import replace
import inspect

import numpy as np
from scipy.optimize import minimize_scalar

from priority_mec.config import Config
from priority_mec.resources import allocate_resources, cpu_closed_form, upload_curve
from priority_mec.solvers import solve


def _links(n, rho=1e9):
    return np.full((n, 4), rho), np.zeros((n, 4))


def test_cpu_closed_form_is_square_root_weighted_not_proportional():
    frequency = cpu_closed_form([1e8, 1e8], [1., 4.], 10e9)
    np.testing.assert_allclose(frequency, [10e9 / 3, 20e9 / 3])
    lagrange = np.array([1., 4.]) * 1e8 / frequency**2
    np.testing.assert_allclose(lagrange, lagrange[0], rtol=1e-13)


def test_cpu_closed_form_uses_cycles_and_priority_once():
    frequency = cpu_closed_form([1e8, 4e8], [1., 1.], 9e9)
    np.testing.assert_allclose(frequency, [3e9, 6e9])
    np.testing.assert_allclose(cpu_closed_form([1e8], [2.], 9e9), [9e9])


def test_upload_derivatives_and_convexity():
    x = np.array([.15, .6])
    d = np.array([.6e6, 1.1e6])
    rho = np.array([2e7, 1e9])
    h = 1e-5
    value, first, second = upload_curve(x, d, rho, 1e7)
    upper = upload_curve(x + h, d, rho, 1e7)[0]
    lower = upload_curve(x - h, d, rho, 1e7)[0]
    np.testing.assert_allclose((upper - lower) / (2 * h), first, rtol=1e-7)
    np.testing.assert_allclose((upper - 2 * value + lower) / h**2, second, rtol=5e-6)
    assert np.all(first < 0) and np.all(second > 0)


def test_independent_uav_bandwidth_and_separate_cpu_satellite_pools():
    cfg = Config()
    data = np.array([.5e6] * 4)
    cycles = data * 500
    weights = np.ones(4)
    rho, offset = _links(4)
    result = allocate_resources(data, cycles, weights, rho, offset, [1, 2, 3, 4], cfg)
    assert result["certified"] and result["feasible"]
    np.testing.assert_allclose(result["bandwidth_hz"][:3], [10e6] * 3, rtol=1e-12)
    np.testing.assert_allclose(result["bandwidth_hz"][3], 100e6, rtol=1e-12)
    np.testing.assert_allclose(result["cpu_hz"][:3], 10e9)
    np.testing.assert_allclose(result["cpu_hz"][3], 50e9)


def test_unconstrained_allocation_is_certified_kkt_optimum():
    cfg = Config()
    data = np.array([.5e6, 1e6, 1.5e6])
    cycles = data * 500
    weight = np.array([2., 1., .5])
    rho, offset = _links(3)
    result = allocate_resources(data, cycles, weight, rho, offset, [1, 1, 2], cfg)
    assert result["status"] == "optimal"
    assert not result["metrics"]["constrained"]
    assert 0 <= result["gap"] < 1e-7
    assert result["lower_bound"] <= result["value"] + 1e-12
    np.testing.assert_allclose(result["cpu_hz"][:2], cpu_closed_form(cycles[:2], weight[:2], 10e9))


def test_active_completion_constraint_against_independent_nested_scalar_oracle():
    cfg = replace(Config(), uav_cpu_hz=1.5e9)
    data = np.array([.5e6, .5e6])
    cycles = np.array([.8e9, .2e9])
    weight = np.array([.1, 2.])
    rho, offset = _links(2)
    result = allocate_resources(data, cycles, weight, rho, offset, [1, 1], cfg, deadline_s=np.ones(2))
    assert result["certified"] and result["feasible"], result
    assert result["metrics"]["constrained"]
    assert result["delay_s"][0] > .99999

    def scalar_oracle(fraction):
        upload = upload_curve(np.array([fraction, 1 - fraction]), data, rho[:, 0], 1e7)[0]
        lower_cpu = cycles / (1 - upload)
        if lower_cpu.sum() > cfg.uav_cpu_hz:
            return 1e5 + lower_cpu.sum() / cfg.uav_cpu_hz
        free_cpu = cpu_closed_form(cycles, weight, cfg.uav_cpu_hz)
        first_cpu = np.clip(free_cpu[0], lower_cpu[0], cfg.uav_cpu_hz - lower_cpu[1])
        assigned = np.array([first_cpu, cfg.uav_cpu_hz - first_cpu])
        return float(weight @ (upload + cycles / assigned))

    oracle = minimize_scalar(scalar_oracle, bounds=(.002, .998), method="bounded",
                             options={"xatol": 1e-12})
    np.testing.assert_allclose(result["value"], oracle.fun, atol=2e-7)
    assert result["lower_bound"] <= oracle.fun + 1e-8
    assert result["gap"] < 1e-6


def test_exclusive_infeasible_and_unresolved_are_different():
    cfg = Config()
    data = np.array([1e6])
    rho, offset = _links(1, 1.)
    result = allocate_resources(data, data * 500, [1.], rho, offset, [1], cfg)
    assert result["status"] == "infeasible" and result["certified_infeasible"]
    assert not result["feasible"]


def test_lower_bound_pruning_cannot_accept_infeasible_assignment():
    cfg = Config()
    data = np.array([1e6, 1e6])
    rho, offset = _links(2)
    result = allocate_resources(data, data * 500, [1., 1.], rho, offset,
                                [1, 1], cfg, incumbent_value=.000001)
    assert result["status"] == "pruned"
    assert result["lower_bound"] >= .000001


def test_potential_descent_is_feasible_strict_and_energy_free():
    cfg = replace(Config(), solver_max_sweeps=5)
    data = np.array([.5e6, .8e6, 1e6, 1.5e6])
    rho, offset = _links(4)
    result = solve(data, data * 500, [2., 1., .8, .5], rho, offset, None, cfg)
    assert np.max(result["delay_s"]) <= 1 + 1e-8
    assert result["metrics"]["resource_certified"]
    for history in result["metrics"]["descent_histories"]:
        assert np.all(np.diff(history["values"]) < -cfg.solver_epsilon)
    assert result["value"] < np.dot([2., 1., .8, .5], data * 500 / 1e9)
    assert "energy" not in inspect.signature(allocate_resources).parameters
    altered = replace(cfg, energy_budget_j=1., uav_joule_per_cycle=1000.)
    other = solve(data, data * 500, [2., 1., .8, .5], rho, offset, None, altered)
    np.testing.assert_array_equal(result["association"], other["association"])
    np.testing.assert_array_equal(result["cpu_hz"], other["cpu_hz"])
    assert result["value"] == other["value"]


def test_unavailable_links_keep_all_local_equilibrium():
    data = np.array([.5e6, 2e6])
    rho, offset = _links(2, 1e-10)
    answer = solve(data, data * 500, [1., 1.], rho, offset, None, Config())
    np.testing.assert_array_equal(answer["association"], [0, 0])
    assert answer["metrics"]["epsilon_equilibrium_certified"]




def test_completion_later_than_slot_but_before_own_deadline_is_allowed():
    cfg = replace(Config(), uav_cpu_hz=.6e9)
    data = np.array([.5e6])
    cycles = np.array([.8e9])
    rho, offset = _links(1)
    result = allocate_resources(data, cycles, [.5], rho, offset, [1], cfg,
                                deadline_s=[2.])
    assert result['certified'] and result['feasible'], result
    assert 1 < result['delay_s'][0] < 2


def test_individual_deadline_changes_active_cpu_allocation():
    cfg = replace(Config(), uav_cpu_hz=1.5e9)
    data = np.array([.5e6, .5e6])
    cycles = np.array([.8e9, .2e9])
    weight = np.array([.1, 2.])
    rho, offset = _links(2)
    unconstrained = allocate_resources(data, cycles, weight, rho, offset, [1, 1], cfg,
                                       deadline_s=[5., 5.])
    constrained = allocate_resources(data, cycles, weight, rho, offset, [1, 1], cfg,
                                     deadline_s=[1., 2.])
    assert constrained['certified'] and constrained['metrics']['constrained'], constrained
    assert constrained['metrics']['active_iterations'] > 0
    assert not constrained['metrics']['node_diagnostics']['1']['fallback_used']
    assert constrained['cpu_hz'][0] > unconstrained['cpu_hz'][0]
    assert abs(constrained['delay_s'][0] - 1) < 1e-7
    assert constrained['metrics']['completion_multipliers'][0] > 0


def test_local_deadlines_can_be_temporarily_ignored_only_for_repair():
    cfg = Config()
    data = np.array([2e6])
    rho, offset = _links(1)
    bad = allocate_resources(data, data * 500, [2.], rho, offset, [0], cfg,
                             deadline_s=[.5])
    assert bad['certified_infeasible']
    repair = allocate_resources(data, data * 500, [2.], rho, offset, [0], cfg,
                                deadline_s=[.5], enforce_local_deadlines=False)
    assert repair['certified']
    assert repair['metrics']['local_deadline_violations'] == [0]


def test_weight_and_deadline_are_independent_inputs():
    data = np.array([2e6])
    rho, offset = _links(1)
    cfg = Config()
    answer = allocate_resources(data, data * 500, [100.], rho, offset, [0], cfg,
                                deadline_s=[1.5])
    assert answer['certified'] and answer['delay_s'][0] == 1


def test_necessary_cpu_capacity_failure_is_certified_not_slsqp_guess():
    cfg = replace(Config(), uav_cpu_hz=1e9)
    data = np.array([.5e6, .5e6])
    rho, offset = _links(2)
    answer = allocate_resources(data, np.array([.8e9, .8e9]), [1., 1.], rho,
                                offset, [1, 1], cfg, deadline_s=[1., 1.])
    assert answer['certified_infeasible']
    assert 'lower bounds' in answer['metrics']['proof']


def test_random_active_set_matches_independent_primal_convex_oracles():
    """Validate different deadlines/weights against a separate primal method."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(28)
    for _ in range(8):
        n = 3
        cfg = replace(Config(), uav_cpu_hz=1.7e9)
        data = rng.uniform(.2, .8, n) * 1e6
        cycles = rng.uniform(.2, .8, n) * 1e9
        rho = rng.uniform(2e7, 1e9, (n, 4))
        equal = np.ones(n) / n
        feasible_time = upload_curve(equal, data, rho[:, 0], 1e7)[0] + cycles / (cfg.uav_cpu_hz * equal)
        deadlines = feasible_time * rng.uniform(1.01, 1.12, n)
        weights = rng.uniform(.1, 4, n)
        answer = allocate_resources(data, cycles, weights, rho, np.zeros((n, 4)),
                                    np.ones(n, int), cfg, deadline_s=deadlines)
        def times(z):
            return upload_curve(z[:n], data, rho[:, 0], 1e7)[0] + cycles / (cfg.uav_cpu_hz * z[n:])
        oracle = minimize(lambda z: weights @ times(z), np.r_[equal, equal], method='SLSQP',
                          bounds=[(1e-9, 1)] * (2*n), constraints=[
                              {'type': 'eq', 'fun': lambda z: np.array([sum(z[:n])-1, sum(z[n:])-1])},
                              {'type': 'ineq', 'fun': lambda z: deadlines-times(z)}],
                          options={'ftol': 1e-11, 'maxiter': 500})
        assert answer['certified'], answer
        assert oracle.success
        np.testing.assert_allclose(answer['value'], oracle.fun, atol=2e-6)
        assert answer['lower_bound'] <= oracle.fun + 1e-7
        assert answer['metrics']['active_iterations'] > 0
        assert not answer['metrics']['node_diagnostics']['1']['fallback_used']
