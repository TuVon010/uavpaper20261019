"""Certified convex allocation of independent aerial-node resource pools.

Each task has its own deadline. A slot is a decision interval, not a
completion bound. No energy variables enter this weighted-delay optimizer.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls

_LN2 = np.log(2.0)


def upload_curve(fraction, data_bit, rho_hz, bandwidth_hz):
    """Fixed-power Shannon upload time and its two bandwidth-fraction derivatives."""
    x = np.maximum(np.asarray(fraction, dtype=float), 1e-14)
    snr = np.asarray(rho_hz, dtype=float) / bandwidth_hz
    z = snr / x
    rate = bandwidth_hz * x * np.log1p(z) / _LN2
    dr = bandwidth_hz * (np.log1p(z) - z / (1.0 + z)) / _LN2
    ddr = -bandwidth_hz * snr**2 / (_LN2 * x * (x + snr)**2)
    duration = np.asarray(data_bit, dtype=float) / rate
    first = -duration * dr / rate
    second = duration * (2.0 * (dr / rate)**2 - ddr / rate)
    return duration, first, second


def cpu_closed_form(cycles, weights, cpu_hz):
    """CPU optimum without active deadlines: F sqrt(w C)/sum sqrt(w C)."""
    score = np.sqrt(np.asarray(cycles, float) * np.asarray(weights, float))
    return float(cpu_hz) * score / score.sum() if score.size else score


def upload_time(fraction, data_bit, rho_hz, bandwidth_hz):
    bandwidth = bandwidth_hz * np.maximum(np.asarray(fraction, float), 1e-14)
    return np.asarray(data_bit) * _LN2 / (bandwidth * np.log1p(np.asarray(rho_hz) / bandwidth))


def _bandwidth_newton(data, rho, weights, bandwidth, tolerance=2e-12, initial=None):
    """Solve the monotone bandwidth KKT equations by equality-constrained Newton.

    Diagonal Hessians eliminate the one-dimensional pool multiplier. The
    support-plane (Frank-Wolfe) gap certifies the resulting convex minimum.
    """
    n = len(data)
    if n == 1:
        return np.ones(1), 0.0, 0
    x = np.sqrt(weights * data) if initial is None else np.array(initial, float)
    x /= x.sum()
    gap = np.inf
    for iteration in range(80):
        t, first, second = upload_curve(x, data, rho, bandwidth)
        grad = weights * first
        gap = max(0.0, float(grad @ x - grad.min()))
        if gap <= tolerance * max(1.0, float(weights @ t)):
            return x, gap, iteration + 1
        hess = np.maximum(weights * second, 1e-25)
        multiplier = -np.sum(grad / hess) / np.sum(1.0 / hess)
        direction = -(grad + multiplier) / hess
        alpha = 1.0
        negative = direction < 0
        if negative.any():
            alpha = min(alpha, .99 * float(np.min(-x[negative] / direction[negative])))
        slope = float(grad @ direction)
        objective = float(weights @ t)
        for _ in range(40):
            trial = np.maximum(x + alpha * direction, 1e-14)
            trial /= trial.sum()
            cost = float(weights @ upload_time(trial, data, rho, bandwidth))
            if cost <= objective + 1e-4 * alpha * slope + 1e-14:
                x = trial
                break
            alpha *= .5
        else:
            break
    first = upload_curve(x, data, rho, bandwidth)[1]
    gap = max(0.0, float((weights * first) @ x - np.min(weights * first)))
    return x, gap, iteration + 1


def _node_allocate(data, cycle, weight, rho, offset, deadline, bandwidth, frequency,
                   tolerance, feasibility_tol):
    """Active-set dual Newton with exact conditional CPU and bandwidth KKT solves.

    mu >= 0 is the task-deadline multiplier. For fixed mu, q=w+mu;
    CPU fractions are sqrt(q C)/sum sqrt(q C). Analytic sensitivity of the
    bandwidth and CPU KKT equations gives the dual Hessian. A primal convex
    fallback is retained only when Newton cannot obtain a certificate.
    """
    n = len(data)
    k = cycle / frequency
    exclusive_tx = upload_time(np.ones(n), data, rho, bandwidth)
    exclusive = offset + exclusive_tx + k
    if np.any(exclusive > deadline + feasibility_tol):
        return {"status": "infeasible", "proof": "exclusive resources insufficient"}
    # Even each task's exclusive bandwidth understates its required CPU.
    minimum_cpu = cycle / (deadline - offset - exclusive_tx)
    if minimum_cpu.sum() > frequency * (1 + feasibility_tol):
        return {"status": "infeasible", "proof": "necessary CPU lower bounds exceed capacity"}
    count_bw = 0

    def inner(mu, start=None):
        nonlocal count_bw
        q = weight + mu
        x, _, iterations = _bandwidth_newton(data, rho, q, bandwidth, initial=start)
        count_bw += iterations
        score = np.sqrt(q * k)
        y = score / score.sum()
        tx, dx, ddx = upload_curve(x, data, rho, bandwidth)
        comp = k / y
        dy = -k / y**2
        ddy = 2 * k / y**3
        times = offset + tx + comp
        gx, gy = q * dx, q * dy
        fw = max(0., float(gx @ x - gx.min())) + max(0., float(gy @ y - gy.min()))
        dual = float(q @ times - mu @ deadline)
        lower = dual - fw
        value = float(weight @ times)
        # Hessian of the minimized Lagrangian w.r.t. task multipliers.
        hessian = np.zeros((n, n))
        for derivative, curvature in ((dx, q * ddx), (dy, q * ddy)):
            inverse = 1 / np.maximum(curvature, 1e-30)
            a = derivative * inverse
            hessian -= np.diag(derivative**2 * inverse)
            hessian += np.outer(a, a) / inverse.sum()
        return {"x": x, "y": y, "times": times, "dx": dx, "dy": dy,
                "dual": dual, "lower": lower, "value": value,
                "gap": max(0., value - lower), "hessian": hessian,
                "mu": mu.copy(), "fw": fw}

    def certified(point):
        return (np.all(point["times"] <= deadline + feasibility_tol) and
                point["gap"] <= tolerance * max(1., abs(point["value"])))

    mu = np.zeros(n)
    point = inner(mu)
    if certified(point):
        point.update(status="optimal", constrained=False, active_iterations=0,
                     bandwidth_newton_iterations=count_bw, fallback_used=False)
        return point
    if np.all(point["times"] <= deadline + feasibility_tol):
        point.update(status="unresolved", constrained=False, active_iterations=0,
                     bandwidth_newton_iterations=count_bw, fallback_used=False)
        return point
    # A feasible objective cannot exceed sum_i w_i d_i; any valid dual lower
    # bound above it is an infeasibility certificate, not an optimizer flag.
    feasible_upper = float(weight @ deadline)
    active_trace = []
    for iteration in range(100):
        gradient = point["times"] - deadline
        active = (mu > 1e-11) | (gradient > feasibility_tol)
        active_trace.append(np.flatnonzero(active).tolist())
        if not active.any():
            break
        # Solve the current active KKT equations, removing zero multipliers
        # whose Newton direction would leave the nonnegative orthant.
        direction = np.zeros(n)
        for _ in range(n + 1):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            hessian = -point["hessian"][np.ix_(indices, indices)]
            ridge = max(1e-13, float(np.max(np.diag(hessian))) * 1e-11)
            try:
                direction[indices] = np.linalg.solve(hessian + ridge * np.eye(len(indices)),
                                                     gradient[indices])
            except np.linalg.LinAlgError:
                direction[indices] = gradient[indices] / np.maximum(np.diag(hessian), ridge)
            reject = active & (mu <= 1e-11) & (direction < 0)
            if not reject.any():
                break
            active[reject] = False
            direction[reject] = 0
        slope = float(gradient @ direction)
        if slope <= 1e-18 or not np.isfinite(direction).all():
            direction = np.maximum(gradient, 0.)
            direction[(mu > 1e-11)] = gradient[(mu > 1e-11)]
            direction /= max(1., float(np.max(np.abs(direction))))
            slope = float(gradient @ direction)
        alpha = 1.
        negative = direction < 0
        if negative.any():
            alpha = min(alpha, float(np.min(-mu[negative] / direction[negative])))
        for _ in range(36):
            candidate_mu = np.maximum(0., mu + alpha * direction)
            candidate = inner(candidate_mu, point["x"])
            if candidate["dual"] >= point["dual"] + 1e-4 * alpha * slope - 1e-12:
                mu, point = candidate_mu, candidate
                break
            alpha *= .5
        else:
            break
        if point["lower"] > feasible_upper + tolerance * max(1., feasible_upper):
            point.update(status="infeasible", proof="dual lower bound exceeds feasible objective upper bound",
                         constrained=True, active_iterations=iteration + 1,
                         bandwidth_newton_iterations=count_bw, active_set_trace=active_trace,
                         fallback_used=False)
            return point
        if certified(point):
            point.update(status="optimal", constrained=True, active_iterations=iteration + 1,
                         bandwidth_newton_iterations=count_bw, active_set_trace=active_trace,
                         fallback_used=False)
            return point
        if np.max(mu) > 1e10:
            break

    # Independent primal fallback: numerical success alone never certifies
    # optimality/infeasibility. The same Lagrangian support bound is checked.
    def evaluate(z):
        tx, dx, _ = upload_curve(z[:n], data, rho, bandwidth)
        y = np.maximum(z[n:], 1e-14)
        comp = k / y
        return offset + tx + comp, dx, -k / y**2

    def objective(z): return float(weight @ evaluate(z)[0])
    def gradient(z):
        _, dx, dy = evaluate(z)
        return np.r_[weight * dx, weight * dy]
    equality = np.zeros((2, 2 * n))
    equality[0, :n] = 1
    equality[1, n:] = 1
    def completion_jacobian(z):
        _, dx, dy = evaluate(z)
        jac = np.zeros((n, 2 * n))
        jac[np.arange(n), np.arange(n)] = -dx
        jac[np.arange(n), n + np.arange(n)] = -dy
        return jac
    optimized = minimize(objective, np.r_[point["x"], point["y"]], jac=gradient,
                         method="SLSQP", bounds=[(1e-12, 1.)] * (2 * n),
                         constraints=[{"type": "eq", "fun": lambda z: equality @ z - 1,
                                       "jac": lambda z: equality},
                                      {"type": "ineq", "fun": lambda z: deadline - evaluate(z)[0],
                                       "jac": completion_jacobian}],
                         options={"ftol": 2e-12, "maxiter": 180})
    z = optimized.x
    times, dx, dy = evaluate(z)
    binding = np.flatnonzero(times >= deadline - 2e-6)
    stationarity = np.zeros((2 * n, len(binding) + 2))
    for column, i in enumerate(binding):
        stationarity[i, column] = dx[i]
        stationarity[n + i, column] = dy[i]
    stationarity[:, len(binding):] = equality.T
    eta = np.zeros(n)
    try:
        multipliers = nnls(stationarity, -gradient(z), maxiter=30 * stationarity.shape[1])[0]
        eta[binding] = multipliers[:len(binding)]
    except (RuntimeError, ValueError):
        pass
    q = weight + eta
    gx, gy = q * dx, q * dy
    fw = max(0., float(gx @ z[:n] - gx.min())) + max(0., float(gy @ z[n:] - gy.min()))
    value = float(weight @ times)
    lower = float(q @ times - eta @ deadline) - fw
    feasible = (np.all(times <= deadline + feasibility_tol) and
                np.max(np.abs(equality @ z - 1)) <= feasibility_tol)
    gap = max(0., value - lower)
    status = "optimal" if feasible and gap <= tolerance * max(1., abs(value)) else "unresolved"
    return {"x": z[:n], "y": z[n:], "times": times, "mu": eta,
            "value": value, "lower": lower, "gap": gap, "status": status,
            "constrained": True, "active_iterations": len(active_trace),
            "active_set_trace": active_trace, "fallback_used": True,
            "slsqp_success": bool(optimized.success), "slsqp_iterations": int(optimized.nit),
            "bandwidth_newton_iterations": count_bw}


def allocate_resources(data_bit, cycles, weights, rho_hz, offset_s, association,
                       cfg, *, incumbent_value=np.inf, deadline_s=None,
                       enforce_local_deadlines=True):
    """Certify a fixed integral association (arbitrary subset size supported).

    deadline_s defaults to 1/weights for compatibility with w=1/d. Explicit
    deadlines are preferred. Local-deadline checks may be disabled only for
    association repair; remote constraints always remain active.
    """
    data = np.asarray(data_bit, float)
    cycle, weight = np.asarray(cycles, float), np.asarray(weights, float)
    rho_matrix, offsets = np.asarray(rho_hz, float), np.asarray(offset_s, float)
    assoc = np.asarray(association, int)
    count = len(data)
    m = int(getattr(cfg, "num_uavs", getattr(cfg, "n_uavs", 3)))
    deadline = 1 / weight if deadline_s is None else np.asarray(deadline_s, float)
    if (cycle.shape != (count,) or weight.shape != (count,) or deadline.shape != (count,) or
            assoc.shape != (count,) or rho_matrix.shape != (count, m + 1) or offsets.shape != rho_matrix.shape):
        raise ValueError("task/link arrays have incompatible shapes")
    if (np.any(data <= 0) or np.any(cycle <= 0) or np.any(weight <= 0) or np.any(deadline <= 0) or
            not all(np.isfinite(a).all() for a in (data, cycle, weight, deadline)) or
            np.any((assoc < 0) | (assoc > m + 1))):
        raise ValueError("positive finite tasks/weights/deadlines and valid association required")
    bandwidth = np.zeros(count)
    cpu = np.full(count, float(cfg.local_cpu_hz))
    delay = cycle / float(cfg.local_cpu_hz)
    active = np.flatnonzero(assoc)
    tolerance = float(getattr(cfg, "solver_certificate_tolerance", 1e-6))
    feasibility_tol = float(getattr(cfg, "solver_feasibility_tolerance", getattr(cfg, "feasibility_tol", 1e-7)))
    local_bad = np.flatnonzero((assoc == 0) & (delay > deadline + feasibility_tol))
    local_value = float(weight[assoc == 0] @ delay[assoc == 0])
    metrics = {"constrained": False, "active_iterations": 0,
               "bandwidth_newton_iterations": 0, "local_deadline_violations": local_bad.tolist(),
               "local_deadlines_enforced": bool(enforce_local_deadlines),
               "completion_multipliers": np.zeros(count).tolist(), "node_diagnostics": {}}
    def result(status, feasible=False, certified=False, lower_bound=-np.inf, gap=np.inf, proof=None):
        value = float(weight @ delay) if np.isfinite(delay).all() else np.inf
        if proof:
            metrics["proof"] = proof
        return {"association": assoc.copy(), "bandwidth_hz": bandwidth.copy(), "cpu_hz": cpu.copy(),
                "delay_s": delay.copy(), "value": value, "lower_bound": float(lower_bound),
                "gap": float(gap), "status": status, "feasible": bool(feasible),
                "certified": bool(certified), "certified_infeasible": status == "infeasible", "metrics": metrics}
    if enforce_local_deadlines and len(local_bad):
        return result("infeasible", proof="selected local task exceeds its own deadline")
    if not len(active):
        return result("optimal", True, True, local_value, 0.)
    node = assoc[active] - 1
    bw_caps = np.asarray([cfg.uav_bandwidth_hz] * m + [cfg.satellite_bandwidth_hz], float)
    cpu_caps = np.asarray([cfg.uav_cpu_hz] * m + [cfg.satellite_cpu_hz], float)
    rho = rho_matrix[active, node]
    offset = offsets[active, node]
    if np.any(rho <= 0) or not np.isfinite(rho).all() or np.any(offset < 0) or not np.isfinite(offset).all():
        return result("infeasible", proof="unavailable selected link")
    cheap_lower = local_value + float(weight[active] @ offset)
    for j in np.unique(node):
        group = active[node == j]
        local_rho = rho[node == j]
        capacity = bw_caps[j]
        aggregate = capacity * np.log1p(local_rho.sum() / capacity) / _LN2
        cpu_lower = float(np.sqrt(weight[group] * cycle[group]).sum()**2 / cpu_caps[j])
        pooled = float(np.sqrt(weight[group] * data[group]).sum()**2 / aggregate)
        wideband = float(np.sum(weight[group] * data[group] * _LN2 / local_rho))
        cheap_lower += cpu_lower + max(pooled, wideband)
    if cheap_lower >= incumbent_value - float(getattr(cfg, "solver_epsilon", 1e-7)):
        return result("pruned", lower_bound=cheap_lower)
    lower = local_value
    unresolved = False
    multipliers = np.zeros(count)
    for j in np.unique(node):
        group = active[node == j]
        answer = _node_allocate(data[group], cycle[group], weight[group], rho[node == j],
                                offset[node == j], deadline[group], bw_caps[j], cpu_caps[j],
                                tolerance, feasibility_tol)
        if answer["status"] == "infeasible":
            return result("infeasible", proof=answer["proof"])
        bandwidth[group] = answer["x"] * bw_caps[j]
        cpu[group] = answer["y"] * cpu_caps[j]
        delay[group] = answer["times"]
        multipliers[group] = answer["mu"]
        lower += answer["lower"]
        unresolved |= answer["status"] != "optimal"
        metrics["constrained"] |= answer["constrained"]
        metrics["active_iterations"] += answer["active_iterations"]
        metrics["bandwidth_newton_iterations"] += answer["bandwidth_newton_iterations"]
        metrics["node_diagnostics"][str(int(j + 1))] = {
            key: value for key, value in answer.items()
            if key not in ("x", "y", "times", "mu", "dx", "dy", "hessian")}
    metrics["completion_multipliers"] = multipliers.tolist()
    metrics["max_completion_violation_s"] = max(0., float(np.max(delay[active] - deadline[active])))
    feasible = bool(np.all(delay[active] <= deadline[active] + feasibility_tol))
    value = float(weight @ delay)
    gap = max(0., value - lower)
    certified = feasible and not unresolved and gap <= tolerance * max(1., abs(value))
    if lower >= incumbent_value - float(getattr(cfg, "solver_epsilon", 1e-7)):
        return result("pruned", feasible, False, lower, gap)
    return result("optimal" if certified else "unresolved", feasible, certified, lower, gap)

